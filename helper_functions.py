def volume(trade_history, amount_bought, amount_sold):
    """
    from trade history in a time range, calculate the trading volume
    """
    total_amount = amount_bought + amount_sold
    trade_old = parser.parse(trade_history[len(trade_history) - 1]['date'])
    trade_new = parser.parse(trade_history[0]['date'])
    trade_diff = (trade_new - trade_old).total_seconds()
    try:
        volume = total_amt / float(trade_diff)
    except:
        volume = 0
    
    return volume

def bought_sold(trade_history, trade):
    fcn_buys = 0
    fcn_bought = 0
    fcn_sells = 0
    fcn_sold = 0
    
    if trade_history[trade]['type'] == 'buy':
        fcn_buys += 1
        fcn_bought += float(trade_history[trade]['amount'])
    #NUMBER OF SELLS AND TOTAL AMOUNT SOLD
    else:
        fcn_sells += 1
        fcn_sold += float(trade_history[trade]['amount'])
        
    return fcn_buys, fcn_bought, fcn_sells, fcn_sold
