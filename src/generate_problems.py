import genetic_instruct as gen
import datasets

NUM_SAMPLES = 500
NUM_COLONIES = 20
NUM_CROSSOVER = 3
CROSSOVER_BATCH_SIZE = 6
SEED_BATCH_SIZE = 15
NUM_CODE_ITERATIONS = 3 # Will produce NUM_CODE_ITERATIONS + 1 solutions

if __name__ == "__main__":
    seed_dataset = datasets.load_dataset("mbpp")
    train_data = seed_dataset['train']
    validation_data = seed_dataset['validation']
    test_data = seed_dataset['test']


    seed_data = [train_data[i]['text'] for i in range(len(train_data))]
    print("Length of seed dataset", len(seed_data))
    problems = gen.genetic_instruct(seed_data, NUM_SAMPLES, NUM_COLONIES,
                                    CROSSOVER_BATCH_SIZE, NUM_CROSSOVER,
                                    SEED_BATCH_SIZE, NUM_CODE_ITERATIONS)

    for (i, p) in enumerate(problems):
        for it in range(NUM_CODE_ITERATIONS + 1):
            with open(f"./problems/problem_{i}_it{it}", "w") as f:
                f.write("QUESTION\n\n" + p[0] + "\n\n" + "SOLUTION/TESTS" +
                        "\n\n" + p[1][min(len(p[1]) - 1, it)])
