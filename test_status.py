"""
SmartBus TN - Test Suite for Crowd Status Engine
Stage 1: Standalone AI People Counter

Validates status.get_status logic at boundary values:
49%, 50%, 79%, 80%, 100%, 110%.
"""

from status import get_status


def run_tests():
    capacity = 100
    test_cases = [
        # (occupancy, expected_status, expected_percent, description)
        (49, "GREEN", 49.0, "49% - Below 50% threshold"),
        (50, "YELLOW", 50.0, "50% - Exactly 50% boundary"),
        (79, "YELLOW", 79.0, "79% - Upper bound of moderate crowd"),
        (80, "RED", 80.0, "80% - High crowd threshold"),
        (100, "RED", 100.0, "100% - Full capacity"),
        (110, "OVERFLOW", 110.0, "110% - Exceeding capacity"),
    ]

    all_passed = True
    print("=" * 60)
    print(" SmartBus TN - Status Engine Test Suite")
    print("=" * 60)

    for occ, expected_status, expected_pct, desc in test_cases:
        actual_status, actual_pct = get_status(occ, capacity)
        is_pass = (actual_status == expected_status) and abs(actual_pct - expected_pct) < 1e-4

        result_str = "[ PASS ]" if is_pass else "[ FAIL ]"
        if not is_pass:
            all_passed = False

        print(
            f"{result_str} {occ}/{capacity} ({actual_pct:.1f}%): "
            f"Expected {expected_status}, Got {actual_status} | {desc}"
        )

        # Assert for strict programmatic validation
        assert actual_status == expected_status, (
            f"Assertion Error: At {occ}/{capacity}, expected status {expected_status} but got {actual_status}"
        )
        assert abs(actual_pct - expected_pct) < 1e-4, (
            f"Assertion Error: At {occ}/{capacity}, expected {expected_pct}% but got {actual_pct}%"
        )

    print("=" * 60)
    if all_passed:
        print("RESULT: ALL 6 TESTS PASSED SUCCESSFULLY!")
    else:
        print("RESULT: SOME TESTS FAILED.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
