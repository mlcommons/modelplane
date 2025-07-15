from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.command_line import compact_sut_list
from modelgauge.ensemble_annotator_set import ENSEMBLE_STRATEGIES
from modelgauge.sut_registry import SUTS


def list_annotators():
    print(compact_sut_list(ANNOTATORS))


def list_suts():
    print(compact_sut_list(SUTS))


def list_ensemble_strategies():
    print(compact_sut_list(ENSEMBLE_STRATEGIES))
