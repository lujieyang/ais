import numpy as np
from sympy.functions.combinatorial.numbers import stirling
from scipy.special import comb
from time import time

class Stirling_Assignments(object):
    def __init__(self, n, k):
        self.n = n
        self.k = k
        if n < k:
            raise ValueError("number of balls n must be more than number of bins k")
        self.precomputed_stirling_numbers = -1*np.ones((n+1,k+1))
        self.precomputed_stirling_numbers[n,k] = stirling(n,k)
        self.nums = np.arange(0,self.n).tolist()
        self.nCr_buffer = -1*np.ones((n,k))
        self.total_assignments = self.precomputed_stirling_numbers[n,k]

    def fast_nCk(self, n, k):
        if n > self.n+1 or k > self.k+1:
            return comb(n,k)
        elif self.nCr_buffer[n,k] < 0:
            self.nCr_buffer[n,k] = comb(n,k)
        return self.nCr_buffer[n,k]

    def fast_stirling(self, n, k):
        if n > self.n+1 or k > self.k+1:
            return stirling(n,k)
        elif self.precomputed_stirling_numbers[n,k] < 0:
            self.precomputed_stirling_numbers[n,k] = stirling(n,k)
        return self.precomputed_stirling_numbers[n,k]

    def get_ith_subset_of_size_m(self, A, i, m):
        n = len(A)
        if m == n:
            return A
        elif m == 0:
            return []
        for j in range(n):
            # number of cases with j as the first element. Compute remaining elements that need to be assigned
            remaining = self.fast_nCk(n - j - 1, m - 1)
            if i < remaining:
                break
            i -= remaining
        return [A[j]] + self.get_ith_subset_of_size_m(A[j + 1:], i, m - 1)

    def gen_part_from_seed(self, A, k, i):
        """
        generate ith unlabelled partition of A into k bins with every bin containing at least 1 element
        """
        if k == 1:
            return [A]
        n = len(A)
        for j in range(0, n - k + 1):
            #remaining assignments to be made if j inside this partition
            remaining = self.fast_stirling(n - 1 - j, k - 1) * self.fast_nCk(n - 1, j)
            if i < remaining:
                break
            i -= remaining
        # We count through the subsets, and for each subset we count through the partitions
        # Split i into a count for subsets and a count for the remaining partitions
        partition_num, subset_num = divmod(i, self.fast_nCk(n - 1, j))
        # Now find the ith appropriate subset
        subset = [A[0]] + self.get_ith_subset_of_size_m(A[1:], subset_num, j)
        S = set(subset)
        return [subset] + self.gen_part_from_seed([a for a in A if a not in S], k - 1, partition_num)

    def generate_ith_parition(self, i):
        return self.gen_part_from_seed(self.nums, self.k, i)

    def _convert_to_D_rep(self, partitions):
        D = np.zeros((self.k, self.n))
        for i, r in enumerate(partitions):
            for c in r:
                D[i,c] = 1
        return D

    def all_partitions_generator(self):
        # note this might not be all that efficient
        i = 0
        while i < self.total_assignments:
            if i % (self.total_assignments//100) == 0:
                print("{}/{}".format(i, self.total_assignments))
            yield self.generate_ith_parition(i)
            i += 1

    def sample_random_partitions(self, num_samples):
        sample_numbers = np.random.randint(0, self.total_assignments, num_samples)
        i = 0
        while i < num_samples:
            yield self.generate_ith_parition(sample_numbers[i])
            i += 1

    def all_partitions_generator_D_rep(self):
        i = 0
        while i < self.total_assignments:
            if i % (self.total_assignments // 100) == 0:
                print("{}/{}".format(i, self.total_assignments))
            yield self._convert_to_D_rep(self.generate_ith_parition(i))
            i += 1

    def sample_random_partitions_D_rep(self, num_samples):
        sample_numbers = np.random.randint(0, self.total_assignments, num_samples)
        i = 0
        while i < num_samples:
            yield self._convert_to_D_rep(self.generate_ith_parition(sample_numbers[i]))
            i += 1


if __name__ == '__main__':
    n = 15
    k = 11
    s = Stirling_Assignments(n,k)
    t0 = time()
    for D in s.all_partitions_generator_D_rep():
        tmp = D
    t1 = time()
    print("Time to generate all parts for ({},{})= {}".format(n,k, t1-t0))


