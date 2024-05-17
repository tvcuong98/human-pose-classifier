import subprocess

def main():
    # Command to execute the Python script with provided parameters
    command = [
        "python3",
        "../kinetic-gan.py",
        "--b1", "0.5",
        "--b2", "0.999",
        "--batch_size", "128",
        "--channels", "2",
        "--checkpoint_interval", "6480",
        "--dataset", "h36m",
        "--latent_dim", "512",
        "--mlp_dim", "8",
        "--lr", "0.0002",
        "--n_classes", "9",
        "--n_cpu", "8",
        "--n_critic", "5",
        "--n_epochs", "10000",
        "--sample_interval", "1620",
        "--t_size", "32",
        "--v_size", "16",
        "--csv_path", "/ske/data/kp_16_cover_modes/mixed/trainmixed.csv"
    ]

    # Run the command
    subprocess.run(command)

if __name__ == "__main__":
    main()
