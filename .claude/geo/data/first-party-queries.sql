-- First-party statistics for "What researchers summarize" (plan.md, prio 6).
-- Read-only, aggregate counts only: no user, IP, email or question text is selected.
-- Run with: heroku pg:psql -a summarizepaper -f .claude/geo/data/first-party-queries.sql
BEGIN TRANSACTION READ ONLY;

-- Q1. Period and size of the dataset
SELECT 'summaries' AS dataset, COUNT(*) AS n, MIN(created)::date AS first, MAX(created)::date AS last
  FROM summarizer_summarypaper WHERE COALESCE(summary, '') <> ''
UNION ALL
SELECT 'summary_runs', COUNT(*), MIN(created)::date, MAX(created)::date FROM summarizer_paperhistory
UNION ALL
SELECT 'assistant_questions', COUNT(*), MIN(created)::date, MAX(created)::date FROM summarizer_aiassistant
UNION ALL
SELECT 'pdf_exports', COUNT(*), MIN(created)::date, MAX(created)::date FROM summarizer_pdfhistory;

-- Q2. Summaries by language and year
SELECT EXTRACT(YEAR FROM created)::int AS year, lang, COUNT(*) AS summaries
  FROM summarizer_summarypaper WHERE COALESCE(summary, '') <> ''
 GROUP BY 1, 2 ORDER BY 1, 2;

-- Q3. Summarized papers by primary arXiv category (top 25)
SELECT p.category, COUNT(DISTINCT p.id) AS papers
  FROM summarizer_summarypaper s JOIN summarizer_arxivpaper p ON p.id = s.paper_id
 WHERE COALESCE(s.summary, '') <> ''
 GROUP BY 1 ORDER BY 2 DESC LIMIT 25;

-- Q4. Summarized papers by arXiv group (part before the dot), for comparison with arXiv's own shares
SELECT split_part(p.category, '.', 1) AS archive, COUNT(DISTINCT p.id) AS papers
  FROM summarizer_summarypaper s JOIN summarizer_arxivpaper p ON p.id = s.paper_id
 WHERE COALESCE(s.summary, '') <> ''
 GROUP BY 1 ORDER BY 2 DESC;

-- Q5. Summary runs by category and year (what people ask to summarize)
SELECT EXTRACT(YEAR FROM h.created)::int AS year, p.category, COUNT(*) AS runs
  FROM summarizer_paperhistory h JOIN summarizer_arxivpaper p ON p.arxiv_id = h.arxiv_id
 GROUP BY 1, 2 HAVING COUNT(*) >= 10 ORDER BY 1, 3 DESC;

-- Q6. Assistant questions and PDF exports by category (top 15 each)
SELECT 'questions' AS kind, p.category, COUNT(*) AS n
  FROM summarizer_aiassistant a JOIN summarizer_arxivpaper p ON p.arxiv_id = a.arxiv_id
 GROUP BY 2 ORDER BY 3 DESC LIMIT 15;
SELECT 'pdf_exports' AS kind, p.category, COUNT(*) AS n
  FROM summarizer_pdfhistory x JOIN summarizer_arxivpaper p ON p.arxiv_id = x.arxiv_id
 GROUP BY 2 ORDER BY 3 DESC LIMIT 15;

-- Q7. How old is a paper when it is first summarized? (days between arXiv publication and first summary)
WITH first_summary AS (
  SELECT p.id, MIN(s.created)::date - p.published_arxiv AS age_days
    FROM summarizer_summarypaper s JOIN summarizer_arxivpaper p ON p.id = s.paper_id
   WHERE COALESCE(s.summary, '') <> '' AND p.published_arxiv IS NOT NULL
   GROUP BY p.id, p.published_arxiv)
SELECT COUNT(*) AS papers,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY age_days) AS median_days,
       ROUND(100.0 * AVG((age_days <= 7)::int), 1) AS pct_within_week,
       ROUND(100.0 * AVG((age_days <= 30)::int), 1) AS pct_within_month,
       ROUND(100.0 * AVG((age_days > 365)::int), 1) AS pct_older_than_year
  FROM first_summary WHERE age_days >= 0;

-- Q8. Votes on summary quality (active votes only)
SELECT COUNT(*) AS votes, ROUND(100.0 * AVG((vote = 1)::int), 1) AS pct_up
  FROM summarizer_vote WHERE active;

-- Q9. Most summarized papers (public arXiv papers, by number of summary runs)
SELECT p.arxiv_id, p.title, p.category, COUNT(*) AS runs
  FROM summarizer_paperhistory h JOIN summarizer_arxivpaper p ON p.arxiv_id = h.arxiv_id
 GROUP BY 1, 2, 3 ORDER BY 4 DESC LIMIT 10;

-- Q10. Categories followed in the daily digest (active subscriptions; categories with fewer than 5 subscribers are hidden)
SELECT COUNT(*) AS active_subscriptions FROM summarizer_digestsubscription WHERE active;
SELECT trim(c) AS category, COUNT(*) AS subscribers
  FROM summarizer_digestsubscription, unnest(string_to_array(categories, ',')) AS c
 WHERE active
 GROUP BY 1 HAVING COUNT(*) >= 5 ORDER BY 2 DESC;

ROLLBACK;
