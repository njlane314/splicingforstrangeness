import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models import HitAssociationTransformer, TrajectoryFittingTransformer
from dataset import HitAssociationDataset, TrajectoryFittingDataset
from config_loader import ConfigLoader

def train_hit_association_model(model, train_loader, val_loader, config, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
    for epoch in range(config.get("num_epochs", 10)):
        model.train()
        train_loss = 0.0
        for sequences, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch"):
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item() * sequences.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{config.get('num_epochs', 10)}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def train_trajectory_fitting_model(model, train_loader, val_loader, config, device):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
    for epoch in range(config.get("num_epochs", 10)):
        model.train()
        train_loss = 0.0
        for sequences, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch"):
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item() * sequences.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{config.get('num_epochs', 10)}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    config = ConfigLoader('config.txt')
    config.print_config()

    root = config.get("root_path", "data")
    u_treename = config.get("u_treename", "u_plane_tree")
    v_treename = config.get("v_treename", "v_plane_tree")
    w_treename = config.get("w_treename", "w_plane_tree")
    batch_size = config.get("batch_size", 32)
    d_model = config.get("d_model", 64)
    num_heads = config.get("num_heads", 4)
    num_encoder_layers = config.get("num_encoder_layers", 2)
    num_particles = config.get("num_particles", 2)
    training_fraction = config.get("training_fraction", 0.75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hit_dataset = HitAssociationDataset(root, u_treename, v_treename, w_treename)
    train_size = int(training_fraction * len(hit_dataset))
    val_size = len(hit_dataset) - train_size

    train_dataset, val_dataset = random_split(hit_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    hit_model = HitAssociationTransformer(num_encoder_layers, d_model, num_heads, input_size=3, num_particles=num_particles).to(device)

    print("Starting training for Hit Association model...")
    train_hit_association_model(hit_model, train_loader, val_loader, config, device)
    torch.save(hit_model.state_dict(), "hit_association_transformer.pth")
    print("Hit Association model training complete and saved as 'hit_association_transformer.pth'.")

    trajectory_dataset = TrajectoryFittingDataset(root)
    trajectory_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    trajectory_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    trajectory_model = TrajectoryFittingTransformer(num_encoder_layers, d_model, num_heads, input_size=3, output_size=3).to(device)
    
    print("Starting training for Trajectory Fitting model...")
    train_trajectory_fitting_model(trajectory_model, trajectory_train_loader, trajectory_val_loader, config, device)
    torch.save(trajectory_model.state_dict(), "trajectory_fitting_transformer.pth")
    print("Trajectory Fitting model training complete and saved as 'trajectory_fitting_transformer.pth'.")
