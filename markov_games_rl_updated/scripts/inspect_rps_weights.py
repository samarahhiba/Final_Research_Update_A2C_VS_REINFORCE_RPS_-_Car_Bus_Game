import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from pathlib import Path

# Import your models
from mg.policy_grad import PolicyNet, ValueNet

# -------- CONFIG --------
OUTPUT_DIR = Path("outputs/rps")

PI1_PATH = OUTPUT_DIR / "a2c_pi1.pt"
PI2_PATH = OUTPUT_DIR / "a2c_pi2.pt"
VALUE_PATH = OUTPUT_DIR / "a2c_value.pt"

N_STATES = 1     # RPS has only 1 state
N_ACTIONS = 3    # Rock, Paper, Scissors


# -------- LOAD MODELS --------
def load_models():
    print("\nLoading RPS models...\n")

    pi1 = PolicyNet(N_STATES, N_ACTIONS)
    pi2 = PolicyNet(N_STATES, N_ACTIONS)
    value_net = ValueNet(N_STATES)

    pi1.load_state_dict(torch.load(PI1_PATH))
    pi2.load_state_dict(torch.load(PI2_PATH))
    value_net.load_state_dict(torch.load(VALUE_PATH))

    pi1.eval()
    pi2.eval()
    value_net.eval()

    print("Models loaded successfully.\n")
    return pi1, pi2, value_net


# -------- PRINT WEIGHTS --------
def print_weights(model, name):
    print(f"\n===== {name} =====")
    for param_name, param in model.named_parameters():
        print(f"{param_name}: shape = {tuple(param.shape)}")


# -------- SHOW POLICY --------
def show_policy(pi, label):
    state = torch.tensor([0])  # only one state in RPS

    with torch.no_grad():
        dist = pi.dist(state)
        probs = dist.probs.squeeze().numpy()

    print(f"\n{label} policy (R, P, S):")
    print(probs)


# -------- SHOW VALUE --------
def show_value(V):
    state = torch.tensor([0])

    with torch.no_grad():
        value = V(state).item()

    print(f"\nValue of RPS state: {value:.4f}")


# -------- MAIN --------
def main():
    pi1, pi2, V = load_models()

    print_weights(pi1, "Policy P1")
    print_weights(pi2, "Policy P2")
    print_weights(V, "Value Network")

    show_policy(pi1, "P1")
    show_policy(pi2, "P2")
    show_value(V)


if __name__ == "__main__":
    main()