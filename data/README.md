# Data Dictionary

This document describes the structure, columns, and known data quality issues for all datasets used in the **Practical Foundations for Question Intelligence** project. All data is synthetic and anonymized for privacy-conscious prototyping.

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. File Descriptions](#2-file-descriptions)
  - [2.1. `learner_questions.csv`](#21-learner_questionscsv)
  - [2.2. `lessons.csv`](#22-lessonscsv)
  - [2.3. `users.csv`](#23-userscsv)
- [3. Data Quality Notes](#3-data-quality-notes)
- [4. Data Processing Outputs](#4-data-processing-outputs)
- [5. Contact](#5-contact)

---

## 1. Overview

All raw data files are located in:

```
data/raw/
```

Processed and cleaned data is written to:

```
data/processed/
```

All paths are configured via environment variables in `.env.example` and `src/config.py`.

---

## 2. File Descriptions

### 2.1. `learner_questions.csv`

**Path:** `data/raw/learner_questions.csv`  
**Description:** Synthetic sample of learner-submitted questions, with lesson and user context.

| Column      | Type      | Description                                                      | Example Value                                   |
|-------------|-----------|------------------------------------------------------------------|-------------------------------------------------|
| id          | String    | Unique question identifier                                       | Q001                                            |
| user_id     | String    | Anonymized user ID (FK to `users.csv`)                           | U001                                            |
| lesson_id   | String    | Lesson identifier (FK to `lessons.csv`)                          | L01                                             |
| text        | String    | Raw question text                                                | What is overfitting?                            |
| timestamp   | DateTime  | When the question was asked (varied formats; see notes)          | 2024-03-02 10:10:55                             |
| label       | String    | Human-assigned or simulated label (optional; may be blank)       | concept, clarification, application, (blank)    |

**Row count:** 80+  
**Primary key:** `id`  
**Foreign keys:** `user_id` → `users.csv`, `lesson_id` → `lessons.csv`

---

### 2.2. `lessons.csv`

**Path:** `data/raw/lessons.csv`  
**Description:** Synthetic lesson metadata for context and trend analysis.

| Column   | Type    | Description                                 | Example Value                  |
|----------|---------|---------------------------------------------|-------------------------------|
| id       | String  | Unique lesson identifier                    | L001                          |
| title    | String  | Lesson title                                | Introduction to Python        |
| topic    | String  | Main topic (may be blank)                   | Programming Basics            |
| date     | Date    | Lesson date (YYYY-MM-DD)                    | 2024-01-10                    |

**Row count:** 15–20  
**Primary key:** `id`

---

### 2.3. `users.csv`

**Path:** `data/raw/users.csv`  
**Description:** Synthetic, anonymized user metadata for privacy-aware analysis.

| Column   | Type    | Description                                 | Example Value   |
|----------|---------|---------------------------------------------|----------------|
| id       | String  | Anonymized user ID                          | U001           |
| role     | String  | User role (e.g., 'learner')                 | learner        |
| cohort   | String  | Cohort or group (may be blank)              | Alpha          |

**Row count:** 40+  
**Primary key:** `id`

---

## 3. Data Quality Notes

### `learner_questions.csv`
- **Missing values:** Some `label` fields are blank (unlabeled questions).
- **Duplicates:** Occasional duplicate `text` values (same question asked by different users or repeated).
- **Timestamp formats:** Inconsistent formats (e.g., `2024-03-01 09:15:00`, `03/01/2024 09:15`, `2024/03/04 09:10`, `2024-03-04T09:16:00`).
- **Empty fields:** Some rows have empty `text` fields.
- **Lesson/user references:** All `lesson_id` and `user_id` values should exist in their respective files, but referential integrity is not strictly enforced in raw data.

### `lessons.csv`
- **Missing values:** Some `topic` fields are blank.
- **Duplicates:** Duplicate `title` entries may exist (e.g., repeated lesson names).

### `users.csv`
- **Duplicates:** Some duplicate `id` values (simulated data entry errors).
- **Missing values:** Occasional missing `cohort` values (blank).
- **Role:** All users are `learner` in this dataset.

---

## 4. Data Processing Outputs

Cleaned and processed data is written to `data/processed/` by the cleaning pipeline (`src/data/clean.py`).  
Key cleaning steps include:
- Standardizing timestamp formats to ISO 8601.
- Removing or flagging duplicate rows.
- Filling or flagging missing values.
- Enforcing referential integrity where possible.
- Anonymizing user data (if not already anonymized).

Processed files retain the same schema as raw files, with additional columns for cleaning status or notes as needed.

---

## 5. Contact

For questions about the data or to report issues, please contact the project maintainer or open an issue in the project repository.

---

**Note:**  
All data is synthetic and intended for educational, non-production use only. No real learner or instructor data is included. Privacy and ethical handling are core to this project’s design.