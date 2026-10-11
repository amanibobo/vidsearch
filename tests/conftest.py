import os

# Tests never reach a real bucket, even from a shell that has sourced .env.
for _name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
    os.environ.pop(_name, None)
