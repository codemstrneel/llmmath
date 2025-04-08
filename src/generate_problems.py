import genetic_instruct as gen
import datasets

NUM_SAMPLES = 1000
NUM_COLONIES = 20
NUM_CROSSOVER = 3
CROSSOVER_BATCH_SIZE = 5
SEED_BATCH_SIZE = 25

if __name__ == "__main__":
    seed_dataset = datasets.load_dataset("mbpp")
    train_data = seed_dataset['train']
    validation_data = seed_dataset['validation']
    test_data = seed_dataset['test']


    seed_data = [train_data[i]['text'] for i in range(len(train_data))]
    print("Length of seed dataset", len(seed_data))
    problems = gen.genetic_instruct(seed_data, NUM_SAMPLES, NUM_COLONIES,
                                    CROSSOVER_BATCH_SIZE, NUM_CROSSOVER, SEED_BATCH_SIZE)

    for (i, p) in enumerate(problems):
        with open(f"./problems/problem_{i}", "w") as f:
            f.write("QUESTION\n\n" + p[0] + "\n\n" + "SOL & TESTS" + "\n\n" + p[1])
