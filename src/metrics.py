def compute_ctx(df, shard_map):

    total_tx = len(df)
    cross_tx = 0

    for _, row in df.iterrows():

        sender = row["From"]
        receiver = row["To"]

        if sender not in shard_map:
            continue

        if receiver not in shard_map:
            continue

        if shard_map[sender] != shard_map[receiver]:
            cross_tx += 1

    ctx = cross_tx / total_tx

    return ctx
