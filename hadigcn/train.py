import torch
import time


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_train_loss = 0.0

    for batch in dataloader:
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device) if hasattr(batch, "edge_attr") else None
        batch_idx = batch.batch.to(device)
        labels = batch.y.to(device)

        optimizer.zero_grad()

        logits = model(x, edge_index, batch_idx, edge_attr)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * labels.size(0)

    return epoch_train_loss / max(len(dataloader), 1)



def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device) if hasattr(batch, "edge_attr") else None
            batch_idx = batch.batch.to(device)
            labels = batch.y.to(device)

            logits = model(x, edge_index, batch_idx, edge_attr)
            loss = criterion(logits, labels)

            total_loss += loss.item() 

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    val_loss = total_loss / max(len(dataloader), 1)

    return val_loss, all_logits, all_labels



def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=50,
    log_interval=1,
):
    history = {
        "train_loss_per_epoch": [],
        "val_loss_per_epoch": [],
        "val_logits_per_epoch": [],
        "val_labels_per_epoch": [],
    }

    model.to(device)

    for epoch in range(1, epochs + 1):
        start = time.time()

        # ---- Train ----
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # ---- Validate ----
        if val_loader is not None:
            val_loss, val_logits, val_labels = validate_one_epoch(
                model, val_loader, criterion, device
            )
        else:
            val_loss, val_logits, val_labels = None, None, None

        # ---- Store history ----
        history["train_loss_per_epoch"].append(train_loss)
        history["val_loss_per_epoch"].append(val_loss)
        history["val_logits_per_epoch"].append(val_logits)
        history["val_labels_per_epoch"].append(val_labels)

        # ---- Logging ----
        elapsed = time.time() - start
        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            if val_loss is not None:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"time: {elapsed:.1f}s"
                )
            else:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"time: {elapsed:.1f}s"
                )

    return {
        "history": history,
        "model": model,
    }
