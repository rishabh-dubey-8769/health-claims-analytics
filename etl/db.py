from sqlalchemy import create_engine
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

def get_engine(secrets):
    # Cloud: use Supabase Pooler URL if available
    if "DATABASE_URL" in secrets:
        url = secrets["DATABASE_URL"]
        # Sanitize URL: remove 'pgbouncer=true' if present
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        query_params.pop('pgbouncer', None)  # remove pgbouncer
        
        new_query = urlencode(query_params, doseq=True)
        cleaned_url = urlunparse(parsed._replace(query=new_query))
        
        return create_engine(cleaned_url, pool_pre_ping=True)

    # Local: use normal Postgres connection from individual secrets
    user = secrets["DB_USER"]
    password = secrets["DB_PASSWORD"]
    host = secrets["POSTGRES_HOST"]
    port = secrets["POSTGRES_PORT"]
    db = secrets["POSTGRES_DB"]

    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)
