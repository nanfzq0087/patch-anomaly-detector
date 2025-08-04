import torch
import torch.nn as nn
import os
import joblib

def train_model(model, train_loader, test_loader, args, machine_id):
    is_torch_model = hasattr(model, "to") and hasattr(model, "parameters")
    save_path = f"checkpoints/{args.model}__{machine_id}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    if not is_torch_model:
        print("🧠 使用传统模型进行训练")
        all_train_data = torch.cat([x[0] if isinstance(x, (list, tuple)) else x for x in train_loader], dim=0)
        model.fit(all_train_data.cpu().numpy())
        joblib.dump(model, save_path)
        print(f"✅ 训练结束，模型已保存至：{save_path}")
        return

    # === 神经网络训练流程 ===
    device = args.device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            optimizer.zero_grad()

            if args.model == 'npsr':
                output = model(x, return_feature=True)
                loss = criterion(output["x_hat"], x) + args.beta * output["kl"]
                
            if args.model == "dcdetector":
                series_list, prior_list = model(x)
                loss_list = [model.kl_loss(s, p).mean() for s, p in zip(series_list, prior_list)]
                loss = torch.stack(loss_list).mean()
            else:
                output = model(x)
                if isinstance(output, dict):
                    if 'x_hat_f' in output and 'x_hat_b' in output:
                        loss = (criterion(output['x_hat_f'], x) + criterion(output['x_hat_b'], x)) / 2
                    elif 'x_hat' in output:
                        loss = criterion(output['x_hat'], x)
                    elif 'x_hat1' in output and 'x_hat21' in output:
                        loss = 0.5 * criterion(output['x_hat1'], x) + 0.5 * criterion(output['x_hat21'], x)
                    else:
                        raise ValueError("Unsupported dict output format.")
                else:
                    loss = criterion(output, x)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ 训练结束，模型已保存至：{save_path}")
