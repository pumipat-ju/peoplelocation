import unittest

from backend.finetune_osnet_balanced import (
    DomainBalancedBatchSampler,
    sampler_source_counts,
    source_ratios,
)


class DomainBalancedFinetuningTests(unittest.TestCase):
    def make_records(self):
        records = []
        source_identities = {"peoplelocation": 3, "market1501": 8, "msmt17": 10}
        for source, identities in source_identities.items():
            for identity in range(identities):
                for sample in range(4):
                    records.append({
                        "dataset_source": source,
                        "identity_key": f"{source}:{identity}",
                        "sample_id": f"{source}-{identity}-{sample}",
                    })
        return records

    def test_sampler_is_deterministic_and_exactly_domain_balanced(self):
        records = self.make_records()
        quotas = {"peoplelocation": 3, "market1501": 3, "msmt17": 4}
        first = DomainBalancedBatchSampler(records, quotas, 2, 5, 17)
        second = DomainBalancedBatchSampler(records, quotas, 2, 5, 17)
        first_batches = list(first)
        self.assertEqual(first_batches, list(second))
        counts = sampler_source_counts(records, first_batches)
        self.assertEqual({
            "peoplelocation": 30, "market1501": 30, "msmt17": 40
        }, counts)
        self.assertEqual({
            "peoplelocation": 0.3, "market1501": 0.3, "msmt17": 0.4
        }, source_ratios(counts))

    def test_sampler_changes_deterministically_by_epoch(self):
        records = self.make_records()
        sampler = DomainBalancedBatchSampler(
            records, {"peoplelocation": 3, "market1501": 3, "msmt17": 4},
            2, 2, 19,
        )
        epoch_zero = list(sampler)
        sampler.set_epoch(1)
        self.assertNotEqual(epoch_zero, list(sampler))

    def test_sampler_does_not_create_files_or_records(self):
        records = self.make_records()
        original_length = len(records)
        sampler = DomainBalancedBatchSampler(
            records, {"peoplelocation": 3, "market1501": 3, "msmt17": 4},
            2, 2, 19,
        )
        list(sampler)
        self.assertEqual(original_length, len(records))


if __name__ == "__main__":
    unittest.main()
