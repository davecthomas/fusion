# file: database.py

import snowflake.connector
from typing import Dict, List
from appconfig import AppConfig


class TheDatabase:
    """
    A class to represent database interactions with Snowflake.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize a Snowflake connection using the config object.
        """
        self.config = config
        self.conn = snowflake.connector.connect(
            account=self.config.SNOWFLAKE_ACCOUNT,
            user=self.config.SNOWFLAKE_USER,
            password=self.config.SNOWFLAKE_PASSWORD,
            warehouse=self.config.SNOWFLAKE_WAREHOUSE,
            database=self.config.SNOWFLAKE_DATABASE,
            schema=self.config.SNOWFLAKE_SCHEMA,
        )

    def get_schema_info(self, table_name: str) -> Dict[str, str]:
        """
        Return a dictionary of {column_name: data_type} for the given table
        by querying Snowflake's information schema.
        """
        query = f"""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name.upper()}'
          AND TABLE_SCHEMA = '{self.config.SNOWFLAKE_SCHEMA.upper()}'
          AND TABLE_CATALOG = '{self.config.SNOWFLAKE_DATABASE.upper()}'
        ORDER BY ORDINAL_POSITION
        """
        cursor = self.conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()

        col_map = {}
        for row in results:
            col_name, data_type = row
            col_map[col_name.lower()] = data_type.upper()

        return col_map

    def get_records(self, table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch actual data records from the specified table, up to 'limit' rows.
        Returns a list of dictionaries, one per row, with column_name -> value.
        """
        query = f"""
        SELECT *
        FROM {self.config.SNOWFLAKE_SCHEMA}.{table_name}
        LIMIT {limit}
        """

        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        # Get the column names from the cursor description
        col_names = [desc[0] for desc in cursor.description]

        # Build a list of dictionaries
        records = []
        for row in rows:
            row_dict = {}
            for col_name, val in zip(col_names, row):
                # You can convert col_name to lowercase if you prefer:
                row_dict[col_name.lower()] = val
            records.append(row_dict)

        cursor.close()
        return records
