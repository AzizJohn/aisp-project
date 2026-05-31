# Day 2 — Clean Baseline

## Setup
- **Model:** Qwen2.5-Coder-1.5B-Instruct (no fine-tuning)
- **Dataset:** Spider dev set, first 50 examples
- **Decoding:** greedy (do_sample=False), max_new_tokens=200
- **Metric:** execution accuracy (compare result sets after running SQL on the DB)

## Results
- **Execution accuracy:** 26/50 = 52.0%
- **SQL parsed & executed without error:** 33/50 = 66.0%
- Of the 33 that ran, 7 returned wrong results; the rest were correct.

## Examples the model got RIGHT

**Correct 1**
- Question: How many singers do we have?
- Gold SQL: `SELECT count(*) FROM singer`
- Predicted: `SELECT COUNT(*) FROM singer`

**Correct 2**
- Question: What is the total number of singers?
- Gold SQL: `SELECT count(*) FROM singer`
- Predicted: `SELECT COUNT(*) FROM singer`

**Correct 3**
- Question: Show name, country, age for all singers ordered by age from the oldest to the youngest.
- Gold SQL: `SELECT name ,  country ,  age FROM singer ORDER BY age DESC`
- Predicted: `SELECT T1.Name ,  T1.Country ,  T1.Age FROM singer AS T1 ORDER BY T1.Age DESC`

## Examples the model got WRONG

**Wrong 1**
- Question: What are the names, countries, and ages for every singer in descending order of age?
- Gold SQL: `SELECT name ,  country ,  age FROM singer ORDER BY age DESC`
- Predicted: `SELECT T1.Name ,  T1.Country ,  T1.Age FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID ORDER BY T1.Age DESC`
- *Why wrong (fill in):* TODO — wrong column? hallucinated table? syntax error? missing JOIN?

**Wrong 2**
- Question: Show the name and the release year of the song by the youngest singer.
- Gold SQL: `SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1`
- Predicted: `SELECT T1.Name ,  T2.Song_Release_year FROM singer AS T1 JOIN concert AS T2 ON T1.Singer_ID = T2.Singer_ID ORDER BY T1.Age ASC LIMIT 1`
- *Why wrong (fill in):* TODO — wrong column? hallucinated table? syntax error? missing JOIN?

**Wrong 3**
- Question: What are the names and release years for all the songs of the youngest singer?
- Gold SQL: `SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1`
- Predicted: `SELECT T1.Name, T2.Song_Release_year FROM singer AS T1 INNER JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID ORDER BY T1.Age ASC LIMIT 1`
- *Why wrong (fill in):* TODO — wrong column? hallucinated table? syntax error? missing JOIN?

## Takeaway
- TODO: 1-2 sentences. The clean model is a reasonable Text-to-SQL baseline at ~52%. Fine-tuning on Spider (Day 3) should raise this. This number is the reference for measuring whether later backdoor fine-tuning harms clean performance.
