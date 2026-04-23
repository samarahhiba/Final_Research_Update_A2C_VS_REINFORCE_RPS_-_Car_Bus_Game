import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from mg.policy_grad import PolicyNet, ValueNet


BASE_DIR = Path("outputs")
DUMP_DIR = Path("weight_dumps")


# -------- UTILS --------

def ensure(path):
    path.mkdir(parents=True, exist_ok=True)


def save_weights(model, filepath):
    with open(filepath, "w") as f:
        for name, param in model.named_parameters():
            arr = param.detach().cpu().numpy()

            f.write(f"\n--- {name} | shape={arr.shape} ---\n")
            f.write(str(arr))
            f.write("\n")


def load_policy(path, n_states, n_actions):
    model = PolicyNet(n_states, n_actions)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def load_value(path, n_states):
    model = ValueNet(n_states)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# -------- MAIN --------

def main():

    # =========================
    # CAR-BUS
    # =========================
    print("Processing Car-Bus...")

    cb_base = DUMP_DIR / "car_bus"
    ensure(cb_base)

    # A2C
    try:
        a2c_dir = cb_base / "a2c"
        ensure(a2c_dir)

        pi1 = load_policy(BASE_DIR / "car_bus/a2c_pi1.pt", 81, 4)
        pi2 = load_policy(BASE_DIR / "car_bus/a2c_pi2.pt", 81, 4)
        V   = load_value(BASE_DIR / "car_bus/a2c_value.pt", 81)

        save_weights(pi1, a2c_dir / "pi1.txt")
        save_weights(pi2, a2c_dir / "pi2.txt")
        save_weights(V,   a2c_dir / "value.txt")

    except Exception as e:
        print("Car-Bus A2C failed:", e)

    # REINFORCE
    try:
        r_dir = cb_base / "reinforce"
        ensure(r_dir)

        pi1 = load_policy(BASE_DIR / "car_bus/reinforce_pi1.pt", 81, 4)
        pi2 = load_policy(BASE_DIR / "car_bus/reinforce_pi2.pt", 81, 4)

        save_weights(pi1, r_dir / "pi1.txt")
        save_weights(pi2, r_dir / "pi2.txt")

    except Exception as e:
        print("Car-Bus REINFORCE failed:", e)


    # =========================
    # RPS
    # =========================
    print("Processing RPS...")

    rps_base = DUMP_DIR / "rps"
    ensure(rps_base)

    # A2C
    try:
        a2c_dir = rps_base / "a2c"
        ensure(a2c_dir)

        pi1 = load_policy(BASE_DIR / "rps/a2c_pi1.pt", 1, 3)
        pi2 = load_policy(BASE_DIR / "rps/a2c_pi2.pt", 1, 3)
        V   = load_value(BASE_DIR / "rps/a2c_value.pt", 1)

        save_weights(pi1, a2c_dir / "pi1.txt")
        save_weights(pi2, a2c_dir / "pi2.txt")
        save_weights(V,   a2c_dir / "value.txt")

    except Exception as e:
        print("RPS A2C failed:", e)

    # REINFORCE
    try:
        r_dir = rps_base / "reinforce"
        ensure(r_dir)

        pi1 = load_policy(BASE_DIR / "rps/reinforce_pi1.pt", 1, 3)
        pi2 = load_policy(BASE_DIR / "rps/reinforce_pi2.pt", 1, 3)

        save_weights(pi1, r_dir / "pi1.txt")
        save_weights(pi2, r_dir / "pi2.txt")

    except Exception as e:
        print("RPS REINFORCE failed:", e)


    print("\n Done! Check weight_dumps/")


if __name__ == "__main__":
    main()