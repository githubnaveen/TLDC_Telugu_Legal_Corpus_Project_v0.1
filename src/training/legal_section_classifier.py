from src.training.classification_baseline import main

if __name__ == "__main__":
    import sys

    if "--task" not in sys.argv:
        sys.argv.extend(["--task", "section_classification"])
    main()
