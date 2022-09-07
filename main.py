import pandas as pd

from lib import absolute_risk_main


def run_snp_only_test():
    model_snp_info = pd.read_csv("./data/bc_72_snps.csv")
    model_disease_incidence_rates = pd.read_csv("./data/model_disease_incidence_rates.csv")
    model_competing_incidence_rates = pd.read_csv("./data/model_competing_incidence_rates.csv")

    absolute_risk_main.compute_absolute_risk(
        model_snp_info=model_snp_info,
        model_disease_incidence_rates=model_disease_incidence_rates,
        model_competing_incidence_rates=model_competing_incidence_rates,
        apply_age_start=50,
        apply_age_interval_length=30,
        return_refs_risk=True
    )


if __name__ == '__main__':
    run_snp_only_test()
