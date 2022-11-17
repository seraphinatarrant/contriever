import shutil
import argparse
import json
import os
from collections import defaultdict

import pandas as pd

all_bolukbasi = ["he","his","her","she","him","man","women","men","woman","spokesman","wife",
                 "himself","son","mother","father","chairman","daughter","husband","guy","girls",
                 "girl","boy","boys","brother","spokeswoman","female","sister","male","herself",
                 "brothers","dad","actress","mom","sons","girlfriend","daughters","lady",
                 "boyfriend","sisters","mothers","king","businessman","grandmother","grandfather",
                 "deer","ladies","uncle","males","congressman","grandson","bull","queen","businessmen",
                 "wives","widow","nephew","bride","females","aunt","prostatecancer","lesbian","chairwoman",
                 "fathers","moms","maiden","granddaughter","youngerbrother","lads","gentleman",
                 "fraternity","bachelor","niece","bulls","husbands","prince","colt","hers",
                 "dude","beard","filly","princess","lesbians","councilman","actresses","gentlemen",
                 "stepfather","monks","exgirlfriend","lad","sperm","testosterone","nephews","maid",
                 "daddy","mare","fiance","fiancee","kings","dads","waitress","maternal","heroine",
                 "nieces","girlfriends","sir","stud","mistress","estrangedwife","womb",
                 "grandma","maternity","estrogen","exboyfriend","widows","gelding","diva",
                 "teenagegirls","nuns","czar","ovariancancer","teenagegirl","penis",
                 "bloke","nun","brides","housewife","spokesmen","suitors","menopause","monastery",
                 "motherhood","stepmother","prostate","hostess","twinbrother",
                 "schoolboy","brotherhood","fillies","stepson","congresswoman","uncles","witch",
                 "monk","paternity","suitor","sorority","businesswoman",
                 "eldestson","gal","statesman","schoolgirl","fathered","goddess","hubby",
                 "stepdaughter","blokes","dudes","strongman","uterus","grandsons","studs","mama",
                 "godfather","hens","hen","mommy","estrangedhusband","elderbrother","boyhood",
                 "baritone","grandmothers","grandpa","boyfriends","feminism","stallion",
                 "heiress","queens","witches","aunts","semen","fella","granddaughters","chap",
                 "widower","convent","vagina","beau","beards","twinsister",
                 "maids","gals","housewives","horsemen","obstetrics","fatherhood","councilwoman",
                 "princes","matriarch","colts","ma","fraternities","pa","fellas","councilmen",
                 "dowry","fraternal","ballerina"]

male_bolukbasi = ["he","his","him","man","men","himself","son","father","husband","guy",
                 "boy","boys","brother","male","brothers","dad","sons","boyfriend","grandfather",
                 "uncle","males","grandson","nephew", "fathers","lads","gentleman",
                 "fraternity","bachelor","husbands","dude","beard","gentlemen","stepfather","lad",
                 "testosterone","nephews","daddy","dads","sir","bloke", "schoolboy","stepson","uncles",
                 "hubby","blokes","dudes","grandsons","godfather","boyhood", "fatherhood",
                 "grandpa","boyfriends","fella","chap","beards","pa","fellas","fraternal", "czar", "baritone",
                 "eldestson", "monks", "elderbrother", "suitor", "stallion", "prostate", "fathered", "youngerbrother",
                 "horsemen", "brotherhood", "bull", "bulls", "businessman", "businessmen", "estrangedhusband", "exboyfriend",
                 "fraternities", "king", "kings", "colt", "colts", "congressman", "chairman", "beau", "councilman", "councilmen"
                 "fatherhood", "gelding", "monk", "monastery", 'paternity','penis','prince','princes', 'prostatecancer', "councilmen",
                 "sperm", "semen", "stud", "studs", "suitors", "spokesman", "spokesmen", "statesman", "strongman", "twinbrother", "widower"]

male_bolukbasi_questionable = ["brethren", "handyman", "countryman", "countrymen", "barbershop", "macho", "lion", "lions", "salesman", "salesmen", "viagra"]

male_extra = ["men's", "mr", "mr.", "emperor"]
female_extra = ["women's", "mrs", "mrs.", "miss", "ms", "ms.", "empress"]

female_bolukbasi = list(set(all_bolukbasi) - set(male_bolukbasi))

# if contains "who" or "person" and no gender terms, is complement

# might want to remove pronouns
female_pronouns = ["she", "her", "herself"] 
male_pronouns = ["he", "his", "him", "himself"]

male_terms = set(male_bolukbasi) | set(male_extra)
female_terms = set(female_bolukbasi) | set(female_extra)
entity_words = set(["who", "whos", "whom", "person", "whose", "name"])


def get_gendered_queries(queries, query_key="text"): 
    # returns all of these, 2 dicts and 4 lists
    male_word2query, female_word2query = defaultdict(list), defaultdict(list)
    gender, complement, male, female = [], [], [], [] 
    for q in queries:
        words = set(q[query_key].lower().split())
        male_words = words & male_terms
        female_words = words & female_terms
        if words & entity_words:
            if not male_words and not female_words:
                complement.append(q)
            else:
                gender.append(q)
                if male_words and female_words: # both genders so don't add to subset
                    continue
                if male_words :
                    male_word2query[male_words.pop()].append(q)
                    male.append(q)
                elif female_words:
                    female_word2query[female_words.pop()].append(q)
                    female.append(q)
    return male_word2query, female_word2query, gender, complement, male, female
    


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', help='dataset to filter')
    p.add_argument('--format', choices=['beir', 'fid'], default='beir')
    p.add_argument('--split', default='test')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()

    target = args.dataset
    
    if args.format == "beir":
        # read in
        data, queries = [], []

        # with open(f"beir/{target}/corpus.jsonl") as fin:
        #     for line in fin:
        #         data.append(json.loads(line))
        with open(f"beir/{target}/queries.jsonl") as fin:
            for line in fin:
                queries.append(json.loads(line))
        with open(f"beir/{target}/qrels/{args.split}.tsv") as fin:
            qrels = pd.read_csv(fin, sep='\t')
        
        # filter
        male_word2query, female_word2query, gender, complement, male, female = get_gendered_queries(queries, query_key="text")
        
        # save
        base = "beir/{}-{}/"
        gender_df = pd.DataFrame(data=gender, columns=['_id','text','metadata'])
        male_df = pd.DataFrame(data=male, columns=['_id','text','metadata'])
        female_df = pd.DataFrame(data=female, columns=['_id','text','metadata'])
        complement_df = pd.DataFrame(data=complement, columns=['_id','text','metadata'])
        print("Example gender queries:")
        print(gender_df.sample(10)["text"].values)
        print("Example complement queries:")
        print(complement_df.sample(10)["text"].values)

        for target_df, name in [(gender_df, "gender"), (complement_df, "complement"), (male_df, "male"), (female_df, "female")]:
            this_base = base.format(target,name)
            os.makedirs(this_base +"qrels/", exist_ok=True)
            qrels_outpath, queries_outpath = this_base +"qrels/test.tsv", this_base + "queries.jsonl"
            target_qrels = qrels[ qrels["query-id"].isin(target_df['_id']) ]

            target_qrels.to_csv(qrels_outpath, sep="\t", index=None)
            target_df.to_json(queries_outpath, lines=True, orient="records")
            # copy train corpus to this base
            shutil.copy2(f"beir/{target}/corpus.jsonl", this_base)
    
    elif args.format == "fid": #  fid datasets are nq and trivia qa (NQ) and (TQA)
        # fields are question answers target ctxs
        with open(f"fid_data/{target}/{args.split}.json", "r") as fin:
            queries = json.load(fin)

        # filter -- this version has everything along with the query but it's simplest to just use the same format as beir where queries are separate. Just only save one file at the end.
        male_word2query, female_word2query, gender, complement, male, female = get_gendered_queries(queries, query_key="question")
    
        # save
        base = f"fid_data/{target}/{args.split}-" + "{}.json"
        cols = ['question', 'answers', 'target', 'ctxs']
        gender_df = pd.DataFrame(data=gender, columns=cols)
        male_df = pd.DataFrame(data=male, columns=cols)
        female_df = pd.DataFrame(data=female, columns=cols)
        complement_df = pd.DataFrame(data=complement, columns=cols)
        print("Example gender queries:")
        print(gender_df.sample(10)["question"].values)
        print("Example complement queries:")
        print(complement_df.sample(10)["question"].values)

        for target_df, name in [(gender_df, "gender"), (complement_df, "complement"), (male_df, "male"), (female_df, "female")]:
            outpath = base.format(name)
            target_df.to_json(outpath, orient="records")
            
    # stats
    print(
        f"num gender: {len(gender)} ({len(gender)/len(queries)*100}%)\n \
        \tnum female: {len(female)} ({len(female)/len(gender)*100}% of gender)\n \
        \tnum male: {len(male)} ({len(male)/len(gender)*100}% of gender)\n \
        num complement: {len(complement)} ({len(complement)/len(queries)*100}%)\n \
        num total: {len(queries)}"
        )
    count_female, count_male = 0, 0
    for p in female_pronouns:
        count_female += len(female_word2query[p])
    for p in male_pronouns:
        count_male += len(male_word2query[p])
    print(f"num gender queries with pronouns (which might be noisy): \
        {count_female + count_male} ({count_female} female {count_male} male)")


