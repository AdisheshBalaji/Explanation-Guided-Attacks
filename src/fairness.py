from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def evaluate_fairness(y_test, y_pred, A_test, model_name="Model"):
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=A_test)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=A_test)
    
    print(f"[{model_name}]")
    print(f"  Demographic Parity Difference : {dp_diff:>+.4f} "
          f"({'✅ PASS' if abs(dp_diff) < 0.10 else '❌ FAIL'})")
    print(f"  Equalized Odds Difference     : {eo_diff:>+.4f} "
          f"({'✅ PASS' if abs(eo_diff) < 0.10 else '❌ FAIL'})")
    print("-" * 60)
    
    return dp_diff, eo_diff