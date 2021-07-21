import os

import pyAgrum as gum


def value_network():
    pe_relative_market_state = "Cheap"
    pe_relative_sector_state = "Cheap"
    forward_pe_current_vs_history_state = "Cheap"
    future_share_performance_state = "Positive"

    ve_model = gum.InfluenceDiagram()

    # Decision node for Expensive_e
    expensive_decision = gum.LabelizedVariable('Expensive_e', '', 2)
    expensive_decision.changeLabel(0, 'No')
    expensive_decision.changeLabel(1, 'Yes')
    ve_model.addDecisionNode(expensive_decision)

    # Decision node for ValueRelativeToPrice
    value_relative_to_price_decision = gum.LabelizedVariable('ValueRelativeToPrice', '', 3)
    value_relative_to_price_decision.changeLabel(0, 'Cheap')
    value_relative_to_price_decision.changeLabel(1, 'FairValue')
    value_relative_to_price_decision.changeLabel(2, 'Expensive')
    ve_model.addDecisionNode(value_relative_to_price_decision)

    # Add a chance node FutureSharePerformance
    future_share_performance = gum.LabelizedVariable('FutureSharePerformance', '', 3)
    future_share_performance.changeLabel(0, 'Positive')
    future_share_performance.changeLabel(1, 'Stagnant')
    future_share_performance.changeLabel(2, 'Negative')
    ve_model.addChanceNode(future_share_performance)

    # Add a chance node PERelative_ShareMarket
    pe_relative_market = gum.LabelizedVariable('PERelative_ShareMarket', '', 3)
    pe_relative_market.changeLabel(0, 'Cheap')
    pe_relative_market.changeLabel(1, 'FairValue')
    pe_relative_market.changeLabel(2, 'Expensive')
    ve_model.addChanceNode(pe_relative_market)

    # Add a chance node PERelative_ShareSector
    pe_relative_sector = gum.LabelizedVariable('PERelative_ShareSector', '', 3)
    pe_relative_sector.changeLabel(0, 'Cheap')
    pe_relative_sector.changeLabel(1, 'FairValue')
    pe_relative_sector.changeLabel(2, 'Expensive')
    ve_model.addChanceNode(pe_relative_sector)

    # Add a chance node ForwardPE_CurrentVsHistory
    forward_pe_current_vs_history = gum.LabelizedVariable('ForwardPE_CurrentVsHistory', '', 3)
    forward_pe_current_vs_history.changeLabel(0, 'Cheap')
    forward_pe_current_vs_history.changeLabel(1, 'FairValue')
    forward_pe_current_vs_history.changeLabel(2, 'Expensive')
    ve_model.addChanceNode(forward_pe_current_vs_history)

    # Utility node for utility_expensive
    utility_expensive = gum.LabelizedVariable('utility_expensive', '', 1)
    ve_model.addUtilityNode(utility_expensive)

    # Utility node for utility_value_relative_to_price
    utility_value_relative_to_price = gum.LabelizedVariable('utility_value_relative_to_price', '', 1)
    ve_model.addUtilityNode(utility_value_relative_to_price)

    # Arcs

    #  ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName(
    # 'PERelative_ShareMarket'))
    # ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName(
    # 'PERelative_ShareSector'))
    # ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName(
    # 'ForwardPE_current_vs_history'))
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('utility_expensive'))

    ve_model.addArc(ve_model.idFromName('PERelative_ShareMarket'), ve_model.idFromName('Expensive_e'))
    ve_model.addArc(ve_model.idFromName('PERelative_ShareMarket'), ve_model.idFromName('ValueRelativeToPrice'))

    ve_model.addArc(ve_model.idFromName('PERelative_ShareSector'), ve_model.idFromName('Expensive_e'))
    ve_model.addArc(ve_model.idFromName('PERelative_ShareSector'), ve_model.idFromName('ValueRelativeToPrice'))
    ve_model.addArc(ve_model.idFromName('PERelative_ShareSector'),
                    ve_model.idFromName('utility_value_relative_to_price'))

    ve_model.addArc(ve_model.idFromName('ForwardPE_CurrentVsHistory'), ve_model.idFromName('ValueRelativeToPrice'))

    ve_model.addArc(ve_model.idFromName('Expensive_e'), ve_model.idFromName('ForwardPE_CurrentVsHistory'))
    ve_model.addArc(ve_model.idFromName('Expensive_e'), ve_model.idFromName('ValueRelativeToPrice'))
    ve_model.addArc(ve_model.idFromName('Expensive_e'), ve_model.idFromName('utility_expensive'))

    ve_model.addArc(ve_model.idFromName('ValueRelativeToPrice'), ve_model.idFromName('utility_value_relative_to_price'))

    # Utilities
    ve_model.utility(ve_model.idFromName('utility_expensive'))[{'Expensive_e': 'Yes'}] = [[100], [75], [-50]]
    # 3 states for FutureSharePerformance node, 2 states for Expensive_e node
    ve_model.utility(ve_model.idFromName('utility_expensive'))[{'Expensive_e': 'No'}] = [[50], [25], [-100]]

    # 3 states for PERelative_ShareSector node, 3 states for ValueRelativeToPrice node
    ve_model.utility(ve_model.idFromName('utility_value_relative_to_price'))[{'ValueRelativeToPrice': 'Cheap'}] = \
        [[-10], [25], [50]]

    ve_model.utility(ve_model.idFromName('utility_value_relative_to_price'))[{'ValueRelativeToPrice': 'FairValue'}] = \
        [[-20], [5], [50]]
    ve_model.utility(ve_model.idFromName('utility_value_relative_to_price'))[{'ValueRelativeToPrice': 'Expensive'}] = \
        [[-50], [25], [100]]

    # CPTs
    # FutureSharePerformance
    if future_share_performance_state == "Positive":
        ve_model.cpt(ve_model.idFromName('FutureSharePerformance'))[0] = 0.986  # Positive
        ve_model.cpt(ve_model.idFromName('FutureSharePerformance'))[1] = 0.9    # Stagnant
        ve_model.cpt(ve_model.idFromName('FutureSharePerformance'))[2] = 0.53   # Negative

    elif future_share_performance_state == "FairValue":
        ve_model.cpt(ve_model.idFromName('FutureSharePerformance'))[1] = 1  # Stagnant
        ve_model.cpt(ve_model.idFromName('FutureSharePerformance'))[0] = 0  # Positive
        ve_model.cpt(ve_model.idFromName('FutureSharePerformance'))[2] = 0  # Negative

    else:
        ve_model.cpt(ve_model.idFromName('FutureSharePerformance'))[2] = 1  # Negative
        ve_model.cpt(ve_model.idFromName('FutureSharePerformance'))[0] = 0  # Positive
        ve_model.cpt(ve_model.idFromName('FutureSharePerformance'))[1] = 0  # Stagnant

    # pe_relative_market
    if pe_relative_market_state == "Cheap":
        ve_model.cpt(ve_model.idFromName('PERelative_ShareMarket'))[0] = 1  # Cheap
        ve_model.cpt(ve_model.idFromName('PERelative_ShareMarket'))[1] = 0  # FairValue
        ve_model.cpt(ve_model.idFromName('PERelative_ShareMarket'))[2] = 0  # Expensive

    elif pe_relative_market_state == "FairValue":
        ve_model.cpt(ve_model.idFromName('PERelative_ShareMarket'))[1] = 1  # FairValue
        ve_model.cpt(ve_model.idFromName('PERelative_ShareMarket'))[0] = 0  # Cheap
        ve_model.cpt(ve_model.idFromName('PERelative_ShareMarket'))[2] = 0  # Expensive

    else:
        ve_model.cpt(ve_model.idFromName('PERelative_ShareMarket'))[2] = 1  # Expensive
        ve_model.cpt(ve_model.idFromName('PERelative_ShareMarket'))[0] = 0  # Cheap
        ve_model.cpt(ve_model.idFromName('PERelative_ShareMarket'))[1] = 0  # FairValue

    # pe_relative_sector
    if pe_relative_sector_state == "Cheap":
        ve_model.cpt(ve_model.idFromName('PERelative_ShareSector'))[0] = 1  # Cheap
        ve_model.cpt(ve_model.idFromName('PERelative_ShareSector'))[1] = 0  # FairValue
        ve_model.cpt(ve_model.idFromName('PERelative_ShareSector'))[2] = 0  # Expensive

    elif pe_relative_sector_state == "FairValue":
        ve_model.cpt(ve_model.idFromName('PERelative_ShareSector'))[1] = 1  # FairValue
        ve_model.cpt(ve_model.idFromName('PERelative_ShareSector'))[0] = 0  # Cheap
        ve_model.cpt(ve_model.idFromName('PERelative_ShareSector'))[2] = 0  # Expensive

    else:
        ve_model.cpt(ve_model.idFromName('PERelative_ShareSector'))[2] = 1  # Expensive
        ve_model.cpt(ve_model.idFromName('PERelative_ShareSector'))[0] = 0  # Cheap
        ve_model.cpt(ve_model.idFromName('PERelative_ShareSector'))[1] = 0  # FairValue

    if forward_pe_current_vs_history_state == "Cheap":
        ve_model.cpt(ve_model.idFromName('ForwardPE_CurrentVsHistory'))[{'Expensive_e': 'Yes'}] = [1, 0, 0]
        ve_model.cpt(ve_model.idFromName('ForwardPE_CurrentVsHistory'))[{'Expensive_e': 'No'}] = [1, 0, 0]

    elif forward_pe_current_vs_history_state == "FairValue":
        ve_model.cpt(ve_model.idFromName('ForwardPE_CurrentVsHistory'))[{'Expensive_e': 'Yes'}] = [0, 1, 0]
        ve_model.cpt(ve_model.idFromName('ForwardPE_CurrentVsHistory'))[{'Expensive_e': 'No'}] = [0, 1, 0]

    else:
        ve_model.cpt(ve_model.idFromName('ForwardPE_CurrentVsHistory'))[{'Expensive_e': 'Yes'}] = [0, 0, 1]
        ve_model.cpt(ve_model.idFromName('ForwardPE_CurrentVsHistory'))[{'Expensive_e': 'No'}] = [0, 0, 1]

    output_file = os.path.join('nets', 'v_e')
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    gum.saveBN(ve_model, os.path.join(output_file, 'v_e.bifxml'))

    ie = gum.ShaferShenoyLIMIDInference(ve_model)
    ie.addNoForgettingAssumption(['Expensive_e', 'ValueRelativeToPrice'])
    ie.makeInference()
    print('--- Inference with default evidence ---')

    # print('Optimal decision for Expensive_e: {0}'.format(ie.optimalDecision('Expensive_e')))
    print('Final decision for Expensive_e: {0}'.format(ie.posterior('Expensive_e')))
    print('Final reward for Expensive_e: {0}'.format(ie.posteriorUtility('Expensive_e')))
    # print('Optimal decision for ValueRelativeToPrice: {0}'.format(ie.optimalDecision('ValueRelativeToPrice')))
    print('Final decision for ValueRelativeToPrice: {0}'.format(ie.posterior('ValueRelativeToPrice')))
    print('Final reward for ValueRelativeToPrice: {0}'.format(ie.posteriorUtility('ValueRelativeToPrice')))
    print('Maximum Expected Utility (MEU) : {0}'.format(ie.MEU()))
    print()
