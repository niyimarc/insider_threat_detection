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


def split_test_partition(test_users, scenario_by_user, dev_frac=0.5, random_state=42):
    """
    Splits the TEST users into a DEVELOPMENT partition and a LOCKED
    partition. Feature design, tuning and ablations are judged on the
    development partition only; the locked partition is evaluated once,
    at the end, so the final numbers were never looked at while
    decisions were being made.

    Insiders are stratified by scenario (each scenario is divided as
    evenly as possible, and any scenario with 2+ insiders puts at least
    one on each side). Benign test users are divided at random by the
    same fraction.

    scenario_by_user: dict user -> scenario for every insider; any test
    user missing from it is treated as benign.
    """
    rng = np.random.default_rng(random_state)
    test_users = sorted(test_users)

    insiders_by_scenario = {}
    benign = []
    for user in test_users:
        if user in scenario_by_user:
            insiders_by_scenario.setdefault(scenario_by_user[user], []).append(user)
        else:
            benign.append(user)

    dev_users = set()
    for scenario in sorted(insiders_by_scenario):
        users = insiders_by_scenario[scenario]
        shuffled = list(rng.permutation(users))
        n_dev = int(round(len(shuffled) * dev_frac))
        if len(shuffled) >= 2:
            n_dev = min(max(n_dev, 1), len(shuffled) - 1)
        dev_users |= set(shuffled[:n_dev])

    shuffled_benign = list(rng.permutation(benign))
    dev_users |= set(shuffled_benign[: int(round(len(shuffled_benign) * dev_frac))])

    locked_users = set(test_users) - dev_users
    return dev_users, locked_users