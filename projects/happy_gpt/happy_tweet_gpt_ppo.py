import copy
import os

import torch
from torch.utils.data.dataloader import DataLoader

from mingpt.logger import Logger
from mingpt.model import GPT
from mingpt.rewards import ValueModel, calculate_advantage_and_returns
from mingpt.trainer import Trainer
from mingpt.utils import set_seed

from happy_tweet_gpt_pg import SentimentRewardModel, TweetDataset


def batch_end_callback(trainer):
    model = trainer.model
    model.eval()

    trainer.logger.log("Train", trainer.iter_num, trainer.loss.item())

    if trainer.iter_num % trainer.config.log_every == 0:
        # evaluate both the train and test score
        with torch.no_grad():
            total_loss = 0
            for i, batch in enumerate(valid_loader):
                x, y, mask = [x.to(trainer.device) for x in batch]
                logits, loss = model(x, y, attention_mask=mask)
                total_loss += loss.item()

        val_loss = total_loss / (i+1)
        trainer.logger.log("Valid", trainer.iter_num, val_loss)
        print(f"E: {trainer.epoch}, iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}, val loss: {val_loss:.5f}")

    if trainer.iter_num % trainer.config.generate_every == 0:
        with torch.no_grad():
            context = train_ds.tokenizer(TEST_PROMPT)[None]
            x = context.to(trainer.device)
            y = model.generate(x, 140, temperature=1.0, do_sample=True, top_k=30)[0]
            completion = train_ds.tokenizer.decode(y)
            print(completion)

    # revert model to training mode
    model.train()


if __name__ == '__main__':
    set_seed(424242)

    print("===== STARTING PRETRAINING =====")

    TEST_PROMPT = "⏎I think "
    valid_iters = 32
    block_size = 32
    train_ds = TweetDataset(block_size, split='train')
    valid_ds = TweetDataset(block_size, split='test', tokenizer=train_ds.tokenizer)
    end_of_text = train_ds.tokenizer.eot_token

    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-micro'
    model_config.vocab_size = train_ds.get_vocab_size()
    model_config.block_size = block_size
    model = GPT(model_config)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 1e-3
    train_config.num_workers = 0
    train_config.log_every = 500
    train_config.generate_every = 1000
    train_config.epochs = 10
    train_config.compile = True
    trainer = Trainer(train_config, model, train_ds)

    sample_size = 3
    device = trainer.device
    sample_prompt = torch.full((sample_size, 1), end_of_text, dtype=torch.long, device=device)

    valid_loader = DataLoader(
        valid_ds,
        shuffle=False,
        batch_size=trainer.config.batch_size * 2,
    )

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    print("\n===== DONE PRETRAINING =====")

    idx = model.generate(sample_prompt, max_new_tokens=80, top_k=30, do_sample=True)
    for j,generation in enumerate(idx):
        print(f"Generation {j}:", train_ds.tokenizer.decode(generation))

    # Plot the losses
    trainer.logger.plot({"Loss": ["Train", "Valid"]}, filename='happy_gpt_pretrain.png')

    print("\n===== STARTING SFT =====")

    sft_dataset = TweetDataset(block_size, split='train', label='positive', tokenizer=train_ds.tokenizer)
    sft_valid = TweetDataset(block_size, split='test', label='positive', tokenizer=train_ds.tokenizer)

    valid_loader = DataLoader(
        sft_valid,
        shuffle=False,
        batch_size=trainer.config.batch_size * 2,
    )

    train_config.learning_rate = 3e-4
    train_config.epochs = 6

    print(len(sft_dataset), "SFT tweets")
    trainer = Trainer(train_config, model, sft_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    print("\n===== DONE SFT =====")
    idx = model.generate(sample_prompt, max_new_tokens=80, top_k=30, do_sample=True)
    for j,generation in enumerate(idx):
        print(f"Generation {j}:", train_ds.tokenizer.decode(generation))

    trainer.logger.plot({"Loss": ["Train", "Valid"]}, filename='happy_gpt_sft.png')

    print("\n===== STARTING RL =====")

    ref_model = copy.deepcopy(model)
    ref_model.requires_grad_(False)
    ref_model.eval()
    reward_model = SentimentRewardModel(train_ds.tokenizer)

    # For ppo we use a value function
    value_model = ValueModel(model_config)
    missing = value_model.load_state_dict(model.state_dict(), strict=False)
    print("Expected some missing value keys:", missing)

    batch_size = 32
    num_iters = 100
    learning_rate = 3e-4
    grad_norm_clip = 1.0
    kl_beta = 0.1
    model.to(device)
    value_model.to(device)
    logger = Logger()

    optim_groups = [{"params": model.parameters()}, {"params": value_model.parameters()}]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
    prompt = torch.full((batch_size, 1), end_of_text, dtype=torch.long, device=device)

    for i in range(num_iters):
        # ------------------------------------------------------------
        # Sample Data
        # These are the `actions` in the RL sense that the model takes
        # ------------------------------------------------------------
        with torch.no_grad():
            model.eval()
            value_model.eval()

            # Sample the prompts
            completion = model.generate(prompt, max_new_tokens=block_size, temperature=1.0, do_sample=True, top_k=30)
            target = completion[:, 1:]
            completion = completion[:, :-1]
            original_log_probs = model.log_probs(completion, target)

            # Reference logprobs
            ref_log_probs = ref_model.log_probs(completion, target)

            # Evaluate the rewards
            rewards = torch.zeros_like(original_log_probs)
            flat_rewards = reward_model(completion)
            rewards[:, -1] = flat_rewards

            # Calculate values, returns and advantages
            values = value_model(completion)

            # Calculate the rewards for each token
            kl = original_log_probs - ref_log_probs
            score = rewards - kl_beta * kl
            advantages, returns = calculate_advantage_and_returns(score, values, torch.ones_like(score), gamma=1.0, lambd=0.95)

        # -------------
        # Train via PPO
        # -------------
        model.train()
        value_model.train()
        # Forward pass through the latest model
        log_probs = model.log_probs(completion, target)

        # Policy loss
        logratio = log_probs - original_log_probs
        ppo_ratio = logratio.exp()
        pg_loss1 = -advantages * ppo_ratio
        pg_loss2 = -advantages * torch.clamp(ppo_ratio, 0.8, 1.2)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        new_value = value_model(completion)
        v_loss = 0.5 * ((new_value - returns) ** 2).mean()

        loss = pg_loss + 0.1 * v_loss

        model.zero_grad(set_to_none=True)
        value_model.zero_grad(set_to_none=True)
        loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        value_grad_norm = torch.nn.utils.clip_grad_norm_(value_model.parameters(), grad_norm_clip)
        optimizer.step()

        # -------
        # Logging
        # -------

        clipfrac = ((ppo_ratio - 1.0).abs() > 0.2).float().mean().item()
        logger.log("Clip Frac", i, clipfrac)
        logger.log("reward", i, flat_rewards.mean().item())
        logger.log("kl", i, kl.mean().item())
        logger.log("Vf loss", i, v_loss.item())

        # Logging
        if i % 10 == 0:
            print(f"Iter: {i}, Avg reward: {flat_rewards.mean().item():.4f}, KL: {kl.mean().item():.4f}, Policy grad norm: {policy_grad_norm:.3f}, Vf Grad norm: {value_grad_norm:.3f}, clip frac: {clipfrac:.3f}")

        if i % 20 == 0:
            model.eval()
            idx = model.generate(sample_prompt, max_new_tokens=80, do_sample=True, top_k=30).cpu()
            for j,generation in enumerate(idx):
                print(f"Generation {j}:", train_ds.tokenizer.decode(generation))
            model.train()

    print("\n===== DONE RL =====")
    idx = model.generate(sample_prompt, max_new_tokens=128, top_k=30, do_sample=True)
    for j,generation in enumerate(idx):
        print(f"Generation {j}:", train_ds.tokenizer.decode(generation))

    print("\n===== SAVING MODEL =====")
    ckpt_path = os.path.join(os.path.curdir, "happy_gpt_ppo.pt")
    torch.save(model.state_dict(), ckpt_path)

    logger.plot({"Reward": ["reward"], "KL Divergence": ["kl"], "Vf Loss": ["Vf loss"]}, filename='happy_gpt_ppo_rl.png')
