"""
One row per student of the Copenhagen Networks Study, built from the call and SMS logs.

Timestamps are seconds from the start of the study; the release does not say at what clock time it
starts, so we use no time-of-day or weekday features. Only counts, durations, contacts and replies.
"""
import numpy as np
import pandas as pd

REPLY_WINDOW_SECONDS = 3600


def call_features(calls):
    rows = {}
    made = calls.groupby("caller")
    received = calls.groupby("callee")
    answered = calls[calls["duration"] >= 0]
    rows["calls_made"] = made.size()
    rows["calls_received"] = received.size()
    rows["calls_missed_made"] = calls[calls["duration"] < 0].groupby("caller").size()
    rows["calls_missed_received"] = calls[calls["duration"] < 0].groupby("callee").size()
    rows["call_seconds_made"] = answered.groupby("caller")["duration"].sum()
    rows["call_seconds_received"] = answered.groupby("callee")["duration"].sum()
    both = pd.concat([answered[["caller", "duration"]].rename(columns={"caller": "user"}),
                      answered[["callee", "duration"]].rename(columns={"callee": "user"})])
    rows["call_median_seconds"] = both.groupby("user")["duration"].median()
    pairs = pd.concat([calls[["caller", "callee"]].set_axis(["user", "other"], axis=1),
                       calls[["callee", "caller"]].set_axis(["user", "other"], axis=1)])
    rows["call_contacts"] = pairs.groupby("user")["other"].nunique()
    days = pd.concat([calls[["caller", "timestamp"]].set_axis(["user", "timestamp"], axis=1),
                      calls[["callee", "timestamp"]].set_axis(["user", "timestamp"], axis=1)])
    rows["call_active_days"] = (days["timestamp"] // 86400).groupby(days["user"]).nunique()
    return pd.DataFrame(rows)


def sms_features(sms):
    rows = {}
    rows["sms_sent"] = sms.groupby("sender").size()
    rows["sms_received"] = sms.groupby("recipient").size()
    pairs = pd.concat([sms[["sender", "recipient"]].set_axis(["user", "other"], axis=1),
                       sms[["recipient", "sender"]].set_axis(["user", "other"], axis=1)])
    rows["sms_contacts"] = pairs.groupby("user")["other"].nunique()
    days = pd.concat([sms[["sender", "timestamp"]].set_axis(["user", "timestamp"], axis=1),
                      sms[["recipient", "timestamp"]].set_axis(["user", "timestamp"], axis=1)])
    rows["sms_active_days"] = (days["timestamp"] // 86400).groupby(days["user"]).nunique()
    rows["sms_reply_share"] = reply_share(sms)
    return pd.DataFrame(rows)


def reply_share(sms, window=REPLY_WINDOW_SECONDS):
    """Share of received messages that the recipient answered (to the same sender) within the window."""
    sent = sms.sort_values("timestamp")
    replies = {}
    by_pair = {key: group["timestamp"].to_numpy() for key, group in sent.groupby(["sender", "recipient"])}
    for (sender, recipient), times in by_pair.items():
        answers = by_pair.get((recipient, sender))
        if answers is None:
            replied = np.zeros(len(times), dtype=bool)
        else:
            # the first answer after each received message
            nxt = np.searchsorted(answers, times, side="right")
            ok = nxt < len(answers)
            replied = np.zeros(len(times), dtype=bool)
            replied[ok] = answers[nxt[ok]] - times[ok] <= window
        got, answered = replies.get(recipient, (0, 0))
        replies[recipient] = (got + len(times), answered + int(replied.sum()))
    return pd.Series({user: answered / got for user, (got, answered) in replies.items()})


def build_people(tables):
    """People with a known gender who appear in the call or SMS log, with their features and label."""
    genders = tables["genders"]
    known = genders[genders["female"].isin([0, 1])].set_index("user")["female"]
    features = call_features(tables["calls"]).join(sms_features(tables["sms"]), how="outer")
    friends = tables.get("fb_friends")
    if friends is not None:
        degree = pd.concat([friends["user_a"], friends["user_b"]]).value_counts()
        features = features.join(degree.rename("facebook_friends"), how="left")
    features = features[features.index.isin(known.index)]
    features = features.fillna(0.0)          # absent from a log = no such activity
    features.index.name = "user"
    people = features.reset_index()
    people.insert(1, "FEMALE", people["user"].map(known).astype(int))
    return people.sort_values("user").reset_index(drop=True)
