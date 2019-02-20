from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination


report_model = BayesianModel([('Tampering', 'Alarm'),
                              ('Fire','Alarm'),
                              ('Alarm', 'Leaving'),
                              ('Leaving', 'Report'),
                              ('Fire', 'Smoke')])

tampering_cpd = TabularCPD(variable='Tampering',
                           variable_card=2,
                           values=[[0.1, 0.9]])

alarm_cpd = TabularCPD(variable='Alarm',
                       variable_card=2,
                       evidence=['Tampering', 'Fire'],
                       evidence_card=[2,2],
                       values=[[0.5, 0.99, 0.85, 0.0001],
                               [0.5, 0.01, 0.15, 0.9999]])

fire_cpd = TabularCPD(variable='Fire',
                      variable_card=2,
                      values=[[0.0001, 0.9999]])

leaving_cpd = TabularCPD(variable='Leaving',
                         variable_card=2,
                         evidence=['Alarm'],
                         evidence_card= [2],
                         values=[[0.88, 0.001],
                                 [0.12, 0.999]])

report_cpd = TabularCPD(variable='Report',
                        variable_card=2,
                        evidence=['Leaving'],
                        evidence_card=[2],
                        values=[[0.75, 0.01],
                                [0.25, 0.99]])

smoke_cpd = TabularCPD(variable='Smoke',
                       variable_card=2,
                       evidence=['Fire'],
                       evidence_card=[2],
                       values=[[0.9, 0.1],
                               [0.1, 0.9]])


report_model.add_cpds(fire_cpd, smoke_cpd, tampering_cpd, alarm_cpd, leaving_cpd, report_cpd)
# print(report_model.get_cpds())
# print(report_model.active_trail_nodes('Report'))
# print(report_model.local_independencies('Alarm'))
# print(report_model.get_independencies())

report_infer = VariableElimination(report_model)
prob_temp = report_infer.query(variables=['Report', 'Leaving'])
# print(prob_temp['Report'])
# print(prob_temp['Leaving'])

prob_alarm_given_smoke_report = report_infer.query(variables=['Alarm'], evidence={'Tampering':0, 'Fire':1})
print(prob_alarm_given_smoke_report['Alarm'])