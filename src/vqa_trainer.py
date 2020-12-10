from vqa_param import args
import json

def main():
    training_args = VQATrainingArguments(
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy=args.eval_strat,
        do_predict=args.do_predict,
        do_train=args.do_train,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        do_eval=args.do_eval,
        weight_decay=0.01,
        label_names=["labels"]
    )

    # May use test for evaluate() to activate wandb logging (cf. predict()).
    trainer = VQATrainer(
        compute_metrics=compute_metrics,
        train_dataset=vqa.train_dataset,
        eval_dataset=eval_set,
        model_init=model_init,
        args=training_args,
    )

    # Trainer with or without hyperparameter search.
    # Save/load best runs if available.
    if training_args.do_train:
        if args.train_type == "grid":
            best = trainer.hyperparameter_search(hp_space=hp_space,
                                          study_name=args.study_name,
                                          n_trials=args.n_trials)
            with open(f'{DATA_PTH}best_run.json', 'w') as br:
                json.dump(best, br)
            print(f"Best run: #{best.run_id}\nObjective: {best.objective}\nHyperparameters: {best.hyperparameters}")
        else:
            if args.load_best:
                try:
                    with open(f"{DATA_PTH}best_run.json") as br:
                        best_run = json.load(br)
                        for n, v in best_run['hyperparameters'].items():
                            setattr(trainer.args, n, v)
                except:
                    print("\nNo best run found.\n")
            trainer.train()

    # Run options for now: run with save_att only OR
    # save_preds and either do_eval only to log eval,
    #                or do_eval and do_predict to log test.

    if training_args.do_eval and not args.do_train:
            eval_ = trainer.evaluate()
            print(f"\nEvaluate:\n{eval_.items()}")

    if training_args.do_predict and not args.do_train:
        if not args.save_preds:
            metrics = trainer.predict(vqa.test_dataset).metrics
            print(f"\nPredict:\n{metrics.items()}")

    # Save attentions to evaluate behavioural similarity (top overlap).
    if args.save_att:
        print("Collecting attentions for eval, test sets.")
        evaluate(trainer, training_args)

    if args.save:
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    from vqa_train_utils import *
    from vqa_paths import DATA_PTH
    main()
