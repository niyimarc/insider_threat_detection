from src.preprocessing.split import split_users, split_test_partition


def test_malicious_users_always_go_to_test():
    all_users = [f"user{i}" for i in range(10)]
    malicious_users = ["user0", "user1"]

    train, val, test = split_users(all_users, malicious_users)

    assert "user0" in test and "user1" in test
    assert "user0" not in train and "user0" not in val
    assert "user1" not in train and "user1" not in val


def test_split_proportions_are_exact_for_clean_numbers():
    all_users = [f"user{i}" for i in range(100)]  # no malicious users, for
                                                    # a clean 100-way split
    train, val, test = split_users(all_users, malicious_users=[])

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_split_is_reproducible_with_same_seed():
    all_users = [f"user{i}" for i in range(50)]
    result_1 = split_users(all_users, malicious_users=[], random_state=42)
    result_2 = split_users(all_users, malicious_users=[], random_state=42)
    assert result_1 == result_2


def test_no_user_appears_in_more_than_one_split():
    all_users = [f"user{i}" for i in range(50)]
    malicious_users = ["user0", "user1", "user2"]

    train, val, test = split_users(all_users, malicious_users)

    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(all_users)


# --- dev / locked partition of the test users ---

def _test_users_and_scenarios():
    insiders = {f"ins{i}": (1 if i < 4 else 2 if i < 10 else 3) for i in range(14)}
    benign = [f"ben{i}" for i in range(20)]
    return set(insiders) | set(benign), insiders


def test_partition_is_disjoint_and_complete():
    test_users, scenarios = _test_users_and_scenarios()
    dev, locked = split_test_partition(test_users, scenarios)
    assert dev & locked == set()
    assert dev | locked == test_users


def test_partition_stratifies_insiders_by_scenario():
    test_users, scenarios = _test_users_and_scenarios()
    dev, locked = split_test_partition(test_users, scenarios)
    for scenario in (1, 2, 3):
        members = {u for u, s in scenarios.items() if s == scenario}
        assert members & dev, f"scenario {scenario} has no insider in dev"
        assert members & locked, f"scenario {scenario} has no insider in locked"


def test_partition_is_reproducible_and_seed_dependent():
    test_users, scenarios = _test_users_and_scenarios()
    assert split_test_partition(test_users, scenarios, random_state=1) == \
        split_test_partition(test_users, scenarios, random_state=1)
    assert split_test_partition(test_users, scenarios, random_state=1) != \
        split_test_partition(test_users, scenarios, random_state=2)


def test_partition_with_single_insider_in_a_scenario_does_not_crash():
    users = {"a", "b", "c", "d"}
    dev, locked = split_test_partition(users, {"a": 1})
    assert dev | locked == users and dev & locked == set()
