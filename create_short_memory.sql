DROP TABLE IF EXISTS [short_memory];
CREATE TABLE [short_memory]
(
    [id] INTEGER NOT NULL PRIMARY KEY,
    [session_id] TEXT NOT NULL,
    [date] DATE NOT NULL,
    [message] TEXT
);