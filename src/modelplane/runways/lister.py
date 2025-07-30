from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.ensemble_annotator_set import ENSEMBLE_STRATEGIES
from modelgauge.sut_registry import SUTS


def list_annotators():
    print(ANNOTATORS.compact_uid_list())


def list_suts():
    print(SUTS.compact_uid_list())


def list_ensemble_strategies():
    print(sorted(ENSEMBLE_STRATEGIES))
