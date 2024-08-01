def make_decision(prediction_results: dict):
    # Initialize counters and lists to track buy and sell probabilities
    buy_track = 0
    sell_track = 0
    prob_b = []
    prob_s = []
    track_b = []
    track_s = []
    prob_total = []

    # Initialize lists to track probabilities for different time frames
    track_b_less10 = []
    track_b_more10 = []
    track_s_less10 = []
    track_s_more10 = []

    # Loop through the prediction_results dictionary
    for k, v in prediction_results.items():
        if k < 4:  # For time frames less than 4 (less than 40 minutes)
            close_less10 = float(v['Close'])
            open_less10 = float(v['Open'])
            diff_less10 = close_less10 - open_less10

            if diff_less10 > 0:
                prob_less10 = diff_less10
                track_b_less10.append(prob_less10)  # Add buy probability to the respective list
            else:
                prob_less10 = diff_less10
                track_s_less10.append(prob_less10)  # Add sell probability to the respective list

        elif k > 10:  # For time frames greater than 10 (more than 100 minutes)
            close_more10 = float(v['Close'])
            open_more10 = float(v['Open'])
            diff_more10 = close_more10 - open_more10

            if diff_more10 > 0:
                prob_more10 = diff_more10
                track_b_more10.append(prob_more10)  # Add buy probability to the respective list
            else:
                prob_more10 = diff_more10
                track_s_more10.append(prob_more10)  # Add sell probability to the respective list

    # Calculate total buy and sell probabilities for different time frames
    total_b_less10 = sum(track_b_less10)
    total_s_less10 = sum(track_s_less10)
    total_b_more10 = sum(track_b_more10)
    total_s_more10 = sum(track_s_more10)

    # Print total probabilities for different time frames
    print('total b less 10: ', total_b_less10)
    print('total s less 10: ', total_s_less10)
    print('total b more 5: ', total_b_more10)
    print('total s less 5: ', total_s_more10)

    # Loop through the prediction_results dictionary again
    for k, v in prediction_results.items():
        close = float(v["Close"])
        openn = float(v["Open"])
        diff = close - openn

        if diff > 0:
            prob = diff
            track_b.append(prob)  # Add buy probability to the respective list
        else:
            prob = diff
            track_s.append(prob)  # Add sell probability to the respective list

    # Calculate total buy and sell probabilities for all time frames
    total_b = sum(track_b)
    total_s = sum(track_s)

    # Print total probabilities for all time frames
    print(total_s)
    print(total_b)

    # Calculate total probability (sum of total buy and total sell)
    total = total_b + total_s
    print(total)

    # Make the final decision based on the total probability
    if total > 0:
        d = str(total) + "B"  # If total is positive, it's a buy decision
    else:
        d = str(total) + "S"  # If total is non-positive, it's a sell decision

    # Print the final decision
    print(f"*********************************  the final decision is:------------->>>> {d}")
