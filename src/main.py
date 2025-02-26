from random_walk import feynman_kac_sample


def bound(x, y):
    return 0


def rhs(x, y):
    return 1


feynman_kac_sample(1000, 0, 0, bound, rhs)
print("ran through")

sample = feynman_kac_sample(1000, .5, .5, bound, rhs)

print(sample)
