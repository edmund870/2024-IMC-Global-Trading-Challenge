# %%
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import collections
from collections import defaultdict
import string
import copy
import pandas as pd
import numpy as np
import math

empty_dict = {'STARFRUIT' : 0, 'AMETHYSTS' : 0, 'ORCHIDS' : 0, 'GIFT_BASKET' : 0, 'CHOCOLATE' : 0, 'STRAWBERRIES' : 0, 'ROSES' : 0}
empty_dict_cache = {'STARFRUIT' : np.array([]), 'AMETHYSTS' : np.array([]), 'ORCHIDS' : np.array([]), 'GIFT_BASKET' : np.array([]), 
                    'CHOCOLATE' : np.array([]), 'STRAWBERRIES' : np.array([]), 'ROSES' : np.array([])}


import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()


class Trader:
    
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'STARFRUIT' : 20, 'AMETHYSTS' : 20, 'ORCHIDS' : 100, 'GIFT_BASKET' : 60, 'CHOCOLATE' : 250, 'STRAWBERRIES' : 350, 'ROSES' : 60}
    spread_cache = copy.deepcopy(empty_dict_cache)
    bid_cache = copy.deepcopy(empty_dict_cache)
    ask_cache = copy.deepcopy(empty_dict_cache)
    mid_cache = copy.deepcopy(empty_dict_cache)
    skew_cache = copy.deepcopy(empty_dict_cache)
    humidity_cache = copy.deepcopy(empty_dict_cache)
    sunlight_cache = copy.deepcopy(empty_dict_cache)
    
    def AMETHYSTS_bs(
          self, 
          best_bid: int, 
          best_ask: int
          ) -> str:
        
        if best_ask <= 10000:       
            return 'buy'
        elif best_bid >= 10000:
            return 'sell'
        else:
            return 'MM'
        
    def STARFRUIT_bs(
          self, 
          best_bid: int, 
          best_ask: int, 
          mid_price_cache: list[float]
          ) -> str:

        product = 'STARFRUIT'
        mid_prices = mid_price_cache[product]

        prev_mid_price = mid_prices[-1] if len(mid_prices) > 1 else 0
        price_spread = best_ask - best_bid

        win = 11
        if len(mid_prices) >= win - 1:
            x = np.arange(1, win, 1)
            y = mid_prices[-(win-1) : ]
            pred_x = win
            coeff = np.polyfit(x, y, deg = 1)
            next_mid = np.polyval(coeff, pred_x)
        else:
            next_mid = prev_mid_price
        
        if best_ask < next_mid and price_spread <= 3:
            return 'buy'
        elif best_bid > next_mid and price_spread <= 3:
            return 'sell'
        else: 
            return 'MM'   
      
    def ORCHIDS_bs(
          self, 
          pos: int,
          best_bid: float,
          best_ask: float,
          mid_price_cache: list[float],
          conv_bid: float,
          conv_ask: float,
          transport_fees: float,
          export_tariff: float,
          import_tariff: float,
          sunlight: list[float],
          humidity: list[float]
    ) -> tuple[str, int, int, int]:
       
       product = 'ORCHIDS'
       storage = 0.1 * 100
       mid_prices = mid_price_cache[product]
       sunlight = sunlight[product]
       humidity = humidity[product]
    
       ############################
       ### Conversion Handling ####
       ############################
       south_buy_price = conv_ask + transport_fees + import_tariff # when buying, we pay transport and receive import tariff (-ve)
       south_sell_price = conv_bid - transport_fees - export_tariff # when selling, we pay transport and export 
        
       conversion = 0
       # if i am long, convert if archipalego sell price less cost of storage (holding for one more period) <= south sell price
       if pos > 0 and best_bid - storage <= south_sell_price: 
           conversion = -pos

       # if short i can convert at the bid of the south
       if pos < 0 and south_buy_price < best_ask: 
           conversion = -pos

       ##############################
       ### Conditions Generation ####
       ##############################
       prev_mid_price = mid_prices[-1] if len(mid_prices) > 1 else 0
       bid_spike = best_bid > prev_mid_price
       ask_spike = best_ask < prev_mid_price
       price_spread = (best_ask - best_bid) / 2

       # when MM, i want to quote ask as close to curr mid price so i get filled (go short) and can convert to gain the import tariff (buying from south)
       ask_spread_adj = int(min(price_spread, abs(south_sell_price - (best_ask - storage))) + 2) 

       # when MM, i want to quote bid as close to curr mid price so i get filled and can convert to earn spread between south buy
       bid_spread_adj = int(abs(south_buy_price - best_bid) + 3)

       if best_ask + storage < south_sell_price:
           cond = 'buy' 
       elif best_bid > south_buy_price:
           cond = 'sell'
       else: 
           cond = 'do nth'

       return cond, conversion, bid_spread_adj, ask_spread_adj


    def GIFT_BASKET_bs(
        self,
        pos: int,
        order_depth: dict[int, int],
        trade_product: str
    ) -> tuple[str, int, int]:

        product = ['GIFT_BASKET', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES']
        product_mids = {}
        worst_bid_dict = {}
        worst_ask_dict = {}

        for prod in product:
            best_ask, _ = list(order_depth[prod].sell_orders.items())[0]
            best_bid, _ = list(order_depth[prod].buy_orders.items())[0]
            worst_ask, _ = next(reversed(order_depth[prod].sell_orders.items()))
            worst_bid, _ = next(reversed(order_depth[prod].buy_orders.items()))
            mid_price = (best_bid + best_ask) / 2
            product_mids[prod] = mid_price
            worst_bid_dict[prod] = worst_bid
            worst_ask_dict[prod] = worst_ask

        component_basket = 4 * product_mids['CHOCOLATE'] + 6 * product_mids['STRAWBERRIES'] + product_mids['ROSES']
        fv = product_mids['GIFT_BASKET'] - component_basket - 387 - 75 / 2

        if trade_product == 'STRAWBERRIES':
            etf_weight =  product_mids['STRAWBERRIES'] / (product_mids['GIFT_BASKET'] - 387 - 75 / 2)
            component_weight = product_mids['STRAWBERRIES'] / component_basket
            fv = component_weight - etf_weight - 0.000062

        if fv < 0: # basket undervalued, buy
            cond = 'buy'
        elif fv > 0: # basket overvalued, sell
            cond = 'sell'
        else:
            cond = 'do nth'
        
        return cond, worst_bid_dict['GIFT_BASKET'], worst_ask_dict['GIFT_BASKET']
        
    def BASKET_COMPONENTS_bs(
        self,
        pos: int,
        order_depth: dict[int, int],
        trade_product: str
    ) -> str:
        fv_threshold = 180 if trade_product == 'ROSES' else -28
        trade_at = fv_threshold
        close_at = fv_threshold * 0.75
        product = ['ROSES', 'CHOCOLATE']
        product_mids = {}
        worst_bid_dict = {}
        worst_ask_dict = {}
        cond = 'do nth'

        for prod in product:
            best_ask, _ = list(order_depth[prod].sell_orders.items())[0]
            best_bid, _ = list(order_depth[prod].buy_orders.items())[0]
            worst_ask, _ = next(reversed(order_depth[prod].sell_orders.items()))
            worst_bid, _ = next(reversed(order_depth[prod].buy_orders.items()))
            mid_price = (best_bid + best_ask) / 2
            product_mids[prod] = mid_price
            worst_bid_dict[prod] = worst_bid
            worst_ask_dict[prod] = worst_ask

        if trade_product == 'ROSES':
            fv = product_mids['ROSES'] - (1.8327 * product_mids['CHOCOLATE'] - product_mids['ROSES'] + 14500)

        if trade_product == 'CHOCOLATE':
            fv = product_mids['CHOCOLATE'] - 0.5456 * product_mids['ROSES']
        
        if fv < -trade_at:
            cond = 'buy'
        elif fv > trade_at:
            cond = 'sell'
        elif fv > close_at and pos < 0:
            cond = 'buy'
        elif fv < -close_at and pos > 0:
            cond = 'sell'
        else:
            cond = 'do nth'

        return cond, worst_bid_dict[trade_product], worst_ask_dict[trade_product], fv

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        # print("traderData: " + state.traderData)
        # logger.print("Observations: " + str(state.observations))
        conversions = 0

        trader_data = ""
                          
        for key, val in state.position.items():
          self.position[key] = val

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:    

          # get order data
          order_depth: OrderDepth = state.order_depths[product]

          # Initialize the list of Orders to be sent as an empty list
          orders: List[Order] = []        

          best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
          best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

          ###############################
          ##### Feature Engineering #####
          ###############################
          price_spread = best_ask - best_bid
          mid_price = (best_bid + best_ask) / 2
          skew = np.log(best_ask) - np.log(best_bid) * 100

          ##################
          ## Order Engine ##
          ##################
          curr_pos = self.position[product]
          pos_limit = self.POSITION_LIMIT[product]
          weight = 0.95

          buy_amount = min(pos_limit, min(best_bid_amount, pos_limit - curr_pos))
          sell_amount = min(pos_limit, abs(min(abs(best_ask_amount), -pos_limit - curr_pos)))

          if product == 'STARFRUIT':
            cond = self.STARFRUIT_bs(best_bid, best_ask, self.mid_cache)
            if cond == 'buy':
                buy_price, sell_price = best_ask, best_bid + 1
            if cond == 'sell':
                buy_price, sell_price = best_bid, best_ask - 1
            if cond == 'MM':
                buy_price, sell_price = best_bid + 1, best_ask - 1 

          if product == 'AMETHYSTS':
            cond = self.AMETHYSTS_bs(best_bid, best_ask)
            if cond == 'buy':
                buy_price, sell_price = best_ask, max(9998, best_bid + 1)
            if cond == 'sell':
                buy_price, sell_price = best_bid, min(10002, best_ask - 1)
            if cond == 'MM':
                buy_price, sell_price = best_bid + 1, best_ask - 1  # i quote the bid, CP wants to sell at bid, i am long

          if product == 'ORCHIDS':
            obs = state.observations.conversionObservations[product]
            conversion_bid = obs.bidPrice
            conversion_ask = obs.askPrice
            conversion_import = obs.importTariff
            conversion_export = obs.exportTariff
            conversion_transport = obs.transportFees
            conversion_humidity = obs.humidity
            conversion_sunlight = obs.sunlight

            cond, conversions, mm_bid_adj, mm_ask_adj = self.ORCHIDS_bs(curr_pos, best_bid, best_ask, self.mid_cache, conversion_bid, conversion_ask, 
                                                                        conversion_transport, conversion_export, conversion_import, self.sunlight_cache, self.humidity_cache)

            if cond == 'buy':
                buy_price, sell_price = best_ask, best_bid + mm_bid_adj
            if cond == 'sell':
                buy_price, sell_price = best_bid, best_ask - mm_ask_adj
            if cond == 'MM':
                buy_price, sell_price = best_bid + mm_bid_adj, best_ask - mm_ask_adj 

          if product == 'GIFT_BASKET':
            cond, bid, ask = self.GIFT_BASKET_bs(curr_pos, state.order_depths, product)
            if cond == 'buy':
                buy_price, sell_price = ask, best_bid
            if cond == 'sell':
                buy_price, sell_price = bid, best_ask
            buy_amount, sell_amount = buy_amount + 2, sell_amount - 2

          if product == 'CHOCOLATE':
            cond, bid, ask, fv = self.BASKET_COMPONENTS_bs(curr_pos, state.order_depths, product)
            if cond == 'buy':
                buy_price, sell_price = best_ask, best_bid
            if cond == 'sell':
                buy_price, sell_price = best_bid, best_ask

          if product == 'STRAWBERRIES':
            cond, bid, ask = self.GIFT_BASKET_bs(curr_pos, state.order_depths, product)
            if cond == 'buy':
                buy_price, sell_price = best_ask, best_bid
            if cond == 'sell':
                buy_price, sell_price = best_bid, best_ask
            
          if product == 'ROSES':
            cond, bid, ask, fv = self.BASKET_COMPONENTS_bs(curr_pos, state.order_depths, product)
            if cond == 'buy':
                buy_price, sell_price = ask, best_bid
            if cond == 'sell':
                buy_price, sell_price = bid, best_ask
            
          logger.print(product, 'current pos', curr_pos)
  
          ### BUY ### 
          if cond == 'buy':
            curr_buy_amount = math.ceil((buy_amount * weight) / 2)
            curr_sell_amount = -math.floor(buy_amount * (1 - weight)) if abs(-math.floor(buy_amount * (1 - weight)) + curr_pos ) <= pos_limit else 0
            if product in ['CHOCOLATE', 'ROSES', 'STRAWBERRIES']:
                trade_vol = pos_limit - curr_pos
                if trade_vol > 0:
                    orders.append(Order(product, buy_price, trade_vol)) # fill ask 
            else:
                orders.append(Order(product, buy_price, curr_buy_amount)) # fill ask
                orders.append(Order(product, sell_price, curr_buy_amount)) # quote bid
                orders.append(Order(product, buy_price, curr_sell_amount)) # quote ask

          ### SELL ###
          if cond == 'sell':
            curr_sell_amount = -math.ceil((sell_amount * weight) / 2)
            curr_buy_amount = math.floor(sell_amount * (1 - weight)) if abs(math.floor(sell_amount * (1 - weight)) + curr_pos ) <= pos_limit else 0

            if product in ['CHOCOLATE', 'ROSES', 'STRAWBERRIES']:
                trade_vol = curr_pos + pos_limit
                if trade_vol > 0:
                    orders.append(Order(product, buy_price, -trade_vol)) # fill bid
            else:
                orders.append(Order(product, buy_price, curr_sell_amount)) # fill bid
                orders.append(Order(product, buy_price, curr_buy_amount)) # quote bid
                orders.append(Order(product, sell_price, curr_sell_amount)) # quote ask

          if cond == 'MM':
            orders.append(Order(product, sell_price, -sell_amount)) # quote ask
            orders.append(Order(product, buy_price, buy_amount)) # quote bid          

          #############
          ## Caching ##
          #############
          window = 10
          caches = [self.spread_cache, self.ask_cache, self.bid_cache, self.mid_cache, self.skew_cache, self.humidity_cache, self.sunlight_cache]
          # Appending Cache
          self.spread_cache[product] = np.append(self.spread_cache[product], price_spread)
          self.ask_cache[product] = np.append(self.ask_cache[product], best_ask)
          self.bid_cache[product] = np.append(self.bid_cache[product], best_bid)
          self.mid_cache[product] = np.append(self.mid_cache[product], mid_price)
          self.skew_cache[product] = np.append(self.skew_cache[product], skew)
          if product == 'ORCHIDS':
            self.humidity_cache[product] = np.append(self.humidity_cache[product], conversion_humidity)
            self.sunlight_cache[product] = np.append(self.sunlight_cache[product], conversion_sunlight)

          # Controlling size of cache
          if len(self.spread_cache[product]) > window:
            for i in caches:
                i[product] = i[product][-window:]  
              
          result[product] = orders

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data