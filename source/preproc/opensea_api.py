import requests

import numpy as np
import pandas as pd

# OpenSea API limits 50 items per a request
# See: 
#   [1] https://docs.opensea.io/reference/api-overview 
#   [2] https://levelup.gitconnected.com/how-to-collect-nft-sales-data-using-opensea-api-5a6b9b163f7
#   [3] http://adilmoujahid.com/posts/2021/06/data-mining-meebits-nfts-python-opensea/

OPENSEA_API_URL = "https://api.opensea.io/api/v1/assets"
REQUEST_LIMIT = 50


class OpenSeaAssetsAPI(object):
    """To retrieve nft asset information"""
    def __init__(
        self,
        num_items=20000,
        asset_contract_address="0x7Bd29408f11D2bFC23c34f18275bBf23bB716Bc7"    # Meebits
    ):
        # Declare parameters
        self.num_items = num_items
        self.asset_contract_address = asset_contract_address
        self.num_requests = num_items // REQUEST_LIMIT

    def fetch(self):
        """Fetch nft asset data via API"""
        parsed_asset_info = list()
        for i  in range(self.num_requests):
            query = {
                "token_ids": list(
                    range((i*REQUEST_LIMIT)+1, ((i+1)*REQUEST_LIMIT)+1)
                ),
                "asset_contract_address", self.asset_contract_address,
                "order_direction": "desc",
                "offset": "0",
                "limit": str(REQUEST_LIMIT)
            }

            response = requests.request(
                "GET", OPENSEA_API_URL, params=query
            )
    
            if response.status_code != 200:
                print("Error Happend!")
                break
            else:
                #Getting meebits data
                asset_records = response.json()["assets"]
        
                #Parsing meebits data
                parsed_asset_info += [
                    self.parse_asset_record(asset_record) 
                    for asset_record in asset_records
                ]
        
        return pd.DataFrame(parsed_asset_info)

    def parse_asset_record(self, asset_record):
        """Return asset information, parsing a single record"""
        # Item information
        item_id = asset_record["token_id"]

        # User information
        try:
            creator_id = asset_record["creator"]["address"]
        except:
            creator_id = None
        finally:
            owner_id = asset_record["owner"]["address"]

        # Transaction information
        traits = asset_record["traits"]
        num_sales = int(asset_record["num_sales"])

        return {
            "item_id": item_id,
            "creator_id": creator_id,
            "owner_id": owner_id,
            "traits": traits,
            "num_sales": num_sales
        }


class OpenSeaEventsAPI(object):
    """To retrieve item transaction information"""
    def __init__(
        self,
        asset_contract_address="0x7Bd29408f11D2bFC23c34f18275bBf23bB716Bc7"    # Meebits
    ):
        # Declare parameters
        self.asset_contract_address = asset_contract_address

    def fetch(self):
        """Fetch nft event data via API"""
        parsed_event_info = list()
        for i  in range(self.num_requests):
            query = {
                "asset_contract_address", self.asset_contract_address,
                "event_type": "successful",
                "only_opensea": "true",
                "offset": str(i*REQUEST_LIMIT),
                "limit": str(REQUEST_LIMIT)
            }

            response = requests.request(
                "GET", OPENSEA_API_URL, params=query
            )

            if response.status_code != 200:
                print("Error Happend!")
                break
            else:
                #Getting meebits data
                event_records = response.json()["asset_events"]

                #Parsing meebits data
                parsed_event_info += [
                    self.parse_event_record(event_record)
                    for event_record in event_records
                ]

        return pd.DataFrame(parsed_event_info)


    def parse_event_record(event_record):
        """Return event infomation, parcing a single record"""
        # Item information
        if event_record["asset"] != None:
            item_id = event_record["asset"]["token_id"]
            b_bundle_event = False
        elif event_record["asset_bundle"] != None:
            item_id = [asset["token_id"] for asset in event_record["asset_bundle"]["assets"]]
            b_bundle_event = True
        
        # User information 
        seller_id = event_record["seller"]["address"]
        buyer_id = event_record["winner_account"]["address"]
        
        # Transaction information
        transaction_hash = event_record["transaction"]["transaction_hash"]
        timestamp = event_record["transaction"]["timestamp"]
        total_price = float(event_record["total_price"])
        payment_token = event_record["payment_token"]["symbol"]
        price_in_usd = float(event_record["payment_token"]["price_in_usd"])

        return {
            "b_bundle_event": b_bundle_event,
            "item_id": item_id,
            "seller_id": seller_id,
            "buyer_id": buyer_id,
            "transaction_hash": transaction_hash,
            "timestamp": timestamp,
            "total_price": total_price,
            "payment_token": payment_token,
            "price_in_usd": price_in_usd,
        }
