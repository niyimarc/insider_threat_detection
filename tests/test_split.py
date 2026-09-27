from src.preprocessing.split import split_users


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