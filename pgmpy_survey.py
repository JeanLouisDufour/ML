from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import PC

reader = BIFReader("bnrepository/survey/survey.bif")
mdl = reader.get_model() # pgmpy.models.BayesianModel

size = 1000
samples = BayesianModelSampling(mdl).forward_sample(size) # pandas.core.frame.DataFrame
samples.to_csv(f'survey_{size}.csv')

est = PC(data=samples) # pgmpy.base.DAG.DAG (-> networkx.classes.digraph.DiGraph)
mdl


