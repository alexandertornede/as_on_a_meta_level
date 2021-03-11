from aslib_scenario.aslib_scenario import ASlibScenario

def save_weights(scenario: ASlibScenario, fold:int, approach, weights):
    print(weights)
    file_name = 'weights/' + scenario.scenario + '.csv'
    with open(file_name, 'a') as f:
        f.write('[{}. {}, {}, {}]\n'.format(str(scenario.scenario), str(fold), str(approach), str(weights).replace('[','').replace(']','')))
        print("Write to file")