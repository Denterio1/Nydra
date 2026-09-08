"""
One-time migration: make users.hashed_password nullable.

Why: Google OAuth users won't have a password at all, but the current
`users` table was created with `hashed_password VARCHAR(255) NOT NULL`.
SQLite can't drop a NOT NULL constraint with ALTER TABLE, so we rebuild
the table: create a new one with the right schema, copy all rows over,
then swap it in.

Safe to run once. Run `check_schema.py` afterward to confirm the change.
"""

import sqlite3

DB_PATH = "nydra.db"


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    try:
        cur.execute("BEGIN TRANSACTION")

        # Sanity check: bail out early if users_new already exists from a
        # previous failed run, instead of silently colliding.
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users_new'"
        )
        if cur.fetchone():
            raise RuntimeError(
                "Table 'users_new' already exists — a previous migration "
                "attempt may have failed partway. Inspect manually before rerunning."
            )

        before_count = cur.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        print(f"Users before migration: {before_count}")

        # 1. Create the new table with hashed_password nullable
        cur.execute(
            """
            CREATE TABLE users_new (
                id VARCHAR NOT NULL,
                username VARCHAR(50) NOT NULL,
                email VARCHAR(255) NOT NULL,
                hashed_password VARCHAR(255),
                is_active BOOLEAN,
                is_admin BOOLEAN,
                created_at DATETIME,
                last_login DATETIME,
                settings JSON,
                google_id VARCHAR(255),
                profile_image VARCHAR(500),
                PRIMARY KEY (id)
            )
            """
        )

        # 2. Copy all existing rows over, column-for-column (explicit list,
        #    not SELECT *, so a future column-order mismatch can't silently
        #    scramble data)
        cur.execute(
            """
            INSERT INTO users_new (
                id, username, email, hashed_password, is_active, is_admin,
                created_at, last_login, settings, google_id, profile_image
            )
            SELECT
                id, username, email, hashed_password, is_active, is_admin,
                created_at, last_login, settings, google_id, profile_image
            FROM users
            """
        )

        after_count = cur.execute("SELECT COUNT(*) FROM users_new").fetchone()[0]
        if after_count != before_count:
            raise RuntimeError(
                f"Row count mismatch after copy: before={before_count}, "
                f"after={after_count}. Aborting — nothing has been committed."
            )

        # 3. Swap tables
        cur.execute("DROP TABLE users")
        cur.execute("ALTER TABLE users_new RENAME TO users")

        # 4. Recreate indexes SQLAlchemy's model expects
        #    (username, email, google_id are all declared with index=True
        #    and/or unique=True in the DBUser model)
        cur.execute("CREATE UNIQUE INDEX ix_users_username ON users (username)")
        cur.execute("CREATE UNIQUE INDEX ix_users_email ON users (email)")
        cur.execute("CREATE UNIQUE INDEX ix_users_google_id ON users (google_id)")

        con.commit()
        print(f"Migration successful. Users after migration: {after_count}")
        print("hashed_password is now nullable. Indexes recreated.")

    except Exception as exc:
        con.rollback()
        print(f"Migration FAILED, rolled back. No changes made. Error: {exc}")
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()