data = pd.read_csv("datasets/amazon_review.csv")
df = data.copy()


#Adım 1 : Total vote' tan helpful_yes (up) değişkenini çıkartarak helpful_no(down) değişkeninin üretilmesi.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


# Adım 2 : score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye eklenmesi

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Adım 3 : İlk 20 yorumu belirleyiniz.

df.sort_values("wilson_lower_bound", ascending=False).head(20)
