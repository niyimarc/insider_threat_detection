import numpy as np


def split_users(all_users, malicious_users, train_frac=0.70, val_frac=0.15, random_state=42):
    """
    Splits users into train/validation/test at the USER level. Every
    malicious user goes to test, regardless of the fractions given. The
    remaining benign users are split according to train_frac/val_frac;
    whatever's left over after that becomes the benign portion of test.
    """
    malicious_users = set(malicious_users)
    benign_users = sorted(set(all_users) - malicious_users)

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(benign_users)

    n_benign = len(shuffled)
    n_train = int(n_benign * train_frac)
    n_val = int(n_benign * val_frac)

    train_users = set(shuffled[:n_train])
    val_users = set(shuffled[n_train:n_train + n_val])
    test_benign_users = set(shuffled[n_train + n_val:])

    test_users = test_benign_users | malicious_users

    return train_users, val_users, test_users