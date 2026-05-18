BEGIN;

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;

CREATE SCHEMA IF NOT EXISTS crypto_app;

CREATE ROLE crypto_auth_admin NOLOGIN;
CREATE ROLE crypto_ingest_writer NOLOGIN;
CREATE ROLE crypto_feature_reader NOLOGIN;
CREATE ROLE crypto_trading_operator NOLOGIN;
CREATE ROLE crypto_audit_viewer NOLOGIN;

GRANT USAGE ON SCHEMA crypto_app TO crypto_auth_admin, crypto_ingest_writer, crypto_feature_reader, crypto_trading_operator, crypto_audit_viewer;

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE crypto_app.app_users TO crypto_auth_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE crypto_app.user_sessions TO crypto_auth_admin;

GRANT INSERT, SELECT ON TABLE crypto_app.raw_market_data TO crypto_ingest_writer;
GRANT INSERT, SELECT ON TABLE crypto_app.raw_onchain_data TO crypto_ingest_writer;
GRANT INSERT, SELECT ON TABLE crypto_app.raw_social_data TO crypto_ingest_writer;
GRANT INSERT, SELECT ON TABLE crypto_app.raw_news_data TO crypto_ingest_writer;

GRANT SELECT ON TABLE crypto_app.feature_views TO crypto_feature_reader;
GRANT SELECT ON TABLE crypto_app.feature_aggregates TO crypto_feature_reader;

GRANT SELECT, INSERT, UPDATE ON TABLE crypto_app.trading_positions TO crypto_trading_operator;

GRANT SELECT ON TABLE crypto_app.trading_audit_events TO crypto_audit_viewer;

ALTER DEFAULT PRIVILEGES IN SCHEMA crypto_app REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA crypto_app GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO crypto_auth_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA crypto_app GRANT INSERT, SELECT ON TABLES TO crypto_ingest_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA crypto_app GRANT SELECT ON TABLES TO crypto_feature_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA crypto_app GRANT SELECT, INSERT, UPDATE ON TABLES TO crypto_trading_operator;
ALTER DEFAULT PRIVILEGES IN SCHEMA crypto_app GRANT SELECT ON TABLES TO crypto_audit_viewer;

COMMIT;
