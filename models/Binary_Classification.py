from pain_recognition.evaluation.binary_classification import (
    PAIRWISE_TASKS,
    run_pairwise_binary_experiment,
)


all_results = []


for baseline_class, pain_class in PAIRWISE_TASKS:

    output = run_pairwise_binary_experiment(

        train_dataframe=
            fused_train,

        validation_dataframe=
            fused_validation,

        test_dataframe=
            fused_test,

        feature_columns=
            feature_columns,

        baseline_class=
            baseline_class,

        pain_class=
            pain_class,

        random_state=42,
    )

    all_results.append(
        output[
            "final_result"
        ]
    )


final_results = pd.DataFrame(
    all_results
)

print(
    final_results
)
