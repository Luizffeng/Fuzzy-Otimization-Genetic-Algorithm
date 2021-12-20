import numpy as np
import skfuzzy as fuzzy
import random as rd
from skfuzzy import control as ctrl
from geneticalgorithm import geneticalgorithm as ga 
import time

last_return = None #global variable for each interation of GA

try:

    ''' Retorna o valor do coeficiente de tensão a ser somado ou diminuido
        Modelo trivial da temperatura'''
    def temp_para_tensao(TDesejado, TMedido):

        if TDesejado >= TMedido:
            if TDesejado - TMedido >= 27: return 1
            else: return (((TDesejado - TMedido)/27) + 1) / 2

        else:
            if TMedido - TDesejado >= 27: return 0
            else: return (((TMedido - TDesejado)/27) * (-1) + 1) / 2


    ''' Implementação da lógica Fuzzy'''
    def chuveiro_fuzzy(parametros_comando):

        # Valores randômicos para serem utilizados, uma vez que os parâmetros ótimos
        # precisam ser calculados para qualquer valor no intervalo do domínio [20, 55]
        TDesejada = rd.randint(20, 55)
        TMedida = rd.randint(20, 55)

        # Definição dos limites
        temp_med = ctrl.Antecedent(np.arange(21, 51, .1), 'temp_med')       # Entrada
        temp_desej = ctrl.Antecedent(np.arange(21, 51, .1), 'temp_desej')   # Entrada
        comando = ctrl.Consequent(np.arange(-0.3, 1.3, .01), 'comando')     # Saída

        temp_med['Fria'] = fuzzy.zmf(temp_med.universe, 23, 37)
        temp_med['Morna'] = fuzzy.trapmf(temp_med.universe, [23, 37, 40, 44])
        temp_med['Quente'] = fuzzy.smf(temp_med.universe, 40, 44)

        temp_desej['Fria'] = fuzzy.zmf(temp_desej.universe, 23, 37)
        temp_desej['Morna'] = fuzzy.trapmf(temp_desej.universe, [23, 37, 40, 44])
        temp_desej['Quente'] = fuzzy.smf(temp_desej.universe, 40, 44)

        comando['Diminuir'] = fuzzy.gaussmf(comando.universe, parametros_comando[2], parametros_comando[3])
        comando['Manter']   = fuzzy.gaussmf(comando.universe, parametros_comando[4], parametros_comando[5])
        comando['Aumentar'] = fuzzy.gaussmf(comando.universe, parametros_comando[6], parametros_comando[7])

        # Criando regras de decisão difusas
        ruleFF = ctrl.Rule(temp_med['Fria'] & temp_desej['Fria'], comando['Manter'])
        ruleFM = ctrl.Rule(temp_med['Fria'] & temp_desej['Morna'], comando['Aumentar'])
        ruleFQ = ctrl.Rule(temp_med['Fria'] & temp_desej['Quente'], comando['Aumentar'])
        ruleMF = ctrl.Rule(temp_med['Morna'] & temp_desej['Fria'], comando['Diminuir'])
        ruleMM = ctrl.Rule(temp_med['Morna'] & temp_desej['Morna'], comando['Manter'])
        ruleMQ = ctrl.Rule(temp_med['Morna'] & temp_desej['Quente'], comando['Aumentar'])
        ruleQF = ctrl.Rule(temp_med['Quente'] & temp_desej['Fria'], comando['Diminuir'])
        ruleQM = ctrl.Rule(temp_med['Quente'] & temp_desej['Morna'], comando['Diminuir'])
        ruleQQ = ctrl.Rule(temp_med['Quente'] & temp_desej['Quente'], comando['Manter'])

        ### Regras de fuzzificação
        comando_control = ctrl.ControlSystem([ruleFF, ruleFM, ruleFQ,
                                            ruleMF, ruleMM, ruleMQ,
                                            ruleQF, ruleQM, ruleQQ])

        comando_simulator = ctrl.ControlSystemSimulation(comando_control)

        # Simulação e Gráfico
        comando_simulator.input['temp_desej'] = TDesejada
        comando_simulator.input['temp_med'] = TMedida
        global last_return

        '''Funções para exibição dos gráficos Fuzzy de entrada e de saída'''
        #temp_med.view(sim=comando_simulator)
        #temp_desej.view(sim=comando_simulator)
        #comando.view(sim=comando_simulator)

        try:
            comando_simulator.compute()
            last_return = comando_simulator.output['comando']
            
        except:
            last_return = last_return

        return (last_return, TDesejada, TMedida)


    ''' Somatória de erros
        Função a ser otimizada pelo GA
        Return the error's sum betwen Fuzzy output and the function "temp_para_tensao" '''
    def soma_erros(parametros):

        x_array = chuveiro_fuzzy(parametros)
        s_fuzzy = x_array[0]
        erro = 0

        for i in range(100):                ### LEVAR AO TOPO DO CÓDIGO PARA PADRONIZAR PARA SER MAIS JUSTO COM CADA INDIVÍDUO, FAZER UM VETOR DE RANDON
            T_desj = rd.randint(20, 55)
            T_med = rd.randint(20, 55)
            erro += (s_fuzzy - temp_para_tensao(T_desj, T_med))**2
        return erro


    ''' Implementação do algoritmo géntico
        com o objetivo de otimizar a função "soma_erros" 
        através da evolução dos parâmetros da função fuzzy'''

    varbound = np.array([[21, 50], [21, 50], [-0.3, 0.3], [-0.3, 0.3], [0.2, 0.8], [-0.3, 0.3], [0.7, 1.3], [-0.3, 0.3]])   # Definição dos limites dos parâmetros
    algorithm_param = {'max_num_iteration': 150, \
                    'population_size': 100, \
                    'mutation_probability': 0.25, \
                    'elit_ratio': 0.15, \
                    'crossover_probability': 0.5, \
                    'parents_portion': 0.3, \
                    'crossover_type': 'uniform', \
                    'max_iteration_without_improv': None}   # Parâmetros géneticos da evolução

    model = ga(function=soma_erros, \
            dimension=8, \
            variable_type='real', \
            variable_boundaries=varbound, \
            algorithm_parameters=algorithm_param)

    model.run()     # Responsável por iniciar o algoritmo genético
    convergence = model.report
    solution = model.output_dict

# Tratamento de erros e ação de parada:
except KeyboardInterrupt:
    print('\n FINALIZADO PELO USUÁRIO')

except Exception as e:
    print(e)

