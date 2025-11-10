# data/data_loader.py

from binance.client import Client
from binance import AsyncClient
import pandas as pd
import time
import asyncio
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union

class DataLoader:
    def __init__(self, client: Client = None, async_client: AsyncClient = None):
        self.client = client
        self.async_client = async_client
        self.logger = logging.getLogger("data_loader")
        self.cache = {}  # Cache simple para evitar llamadas repetidas
        self.cache_timeout = 60  # segundos
        
    def set_clients(self, client: Client, async_client: AsyncClient):
        """Configura ambos clientes (síncrono y asíncrono)"""
        self.client = client
        self.async_client = async_client

    def obtener_ohlcv(self, symbol, interval="1m", limit=100, use_cache=True):
        """
        Versión síncrona mejorada - compatible con tu código existente
        """
        cache_key = f"{symbol}_{interval}_{limit}"
        
        # Verificar cache
        if use_cache and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data.copy()

        try:
            raw = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

            # 🔎 Validar respuesta antes de crear DataFrame
            if not raw or len(raw) == 0:
                print(f"⚠️ Binance devolvió una lista vacía para {symbol}")
                return pd.DataFrame()

            # 🔧 Crear DataFrame completo con todas las columnas
            df = pd.DataFrame(raw, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])

            # 🧩 Validar que todas las columnas existan
            expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(f"⚠️ Columnas faltantes en datos de {symbol}: {missing}")
                return pd.DataFrame()

            # 🧠 Limpieza y conversión segura
            df = df[expected_cols].copy()
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # 🧩 Eliminar duplicados y NaN por seguridad
            df = df.dropna().drop_duplicates()
            if len(df) < 5:
                print(f"⚠️ Muy pocos datos OHLCV ({len(df)}) para {symbol}. Se omite.")
                return pd.DataFrame()

            # 🎯 Formatear para compatibilidad con nuevas estrategias
            df = self._formatear_para_estrategias(df)

            # Guardar en cache
            if use_cache:
                self.cache[cache_key] = (df.copy(), time.time())

            return df

        except Exception as e:
            print(f"❌ Error al obtener OHLCV para {symbol}: {e}")
            return pd.DataFrame()

    async def obtener_ohlcv_async(self, symbol, interval="1m", limit=100, use_cache=True):
        """
        Versión asíncrona para las nuevas estrategias
        """
        cache_key = f"async_{symbol}_{interval}_{limit}"
        
        # Verificar cache
        if use_cache and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data.copy()

        try:
            if not self.async_client:
                return self.obtener_ohlcv(symbol, interval, limit, use_cache)

            # Obtener datos de Binance
            klines = await self.async_client.get_klines(
                symbol=symbol, 
                interval=interval, 
                limit=limit
            )

            if not klines:
                self.logger.warning(f"Datos vacíos para {symbol} {interval}")
                return pd.DataFrame()

           # Procesar datos...
            df = self._procesar_dataframe_async(klines)
            
            if use_cache:
                self.cache[cache_key] = (df.copy(), time.time())

            return df

        except Exception as e:
            self.logger.error(f"Error obteniendo OHLCV async para {symbol}: {e}")
            # Fallback a método síncrono
            return self.obtener_ohlcv(symbol, interval, limit, use_cache)
    
    def _procesar_dataframe_async(self, klines):
        """Procesar datos de la API async"""
        try:
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convertir tipos
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df = df.dropna()
            df = self._formatear_para_estrategias(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error procesando DataFrame async: {e}")
            return pd.DataFrame()

    def obtener_datos_multitimeframe(self, symbol, timeframes=None, limit=100):
        """
        Obtiene datos para múltiples timeframes (síncrono)
        """
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']

        data = {}
        for tf in timeframes:
            try:
                df = self.obtener_ohlcv(symbol, tf, limit)
                if not df.empty:
                    data[tf] = df
                    self.logger.info(f"✅ Datos {tf} obtenidos para {symbol} - {len(df)} registros")
                else:
                    self.logger.warning(f"⚠️ Sin datos {tf} para {symbol}")
            except Exception as e:
                self.logger.error(f"❌ Error obteniendo {tf} para {symbol}: {e}")
                continue

        return data

    async def obtener_datos_multitimeframe_async(self, symbol, timeframes=None, limit=100):
        """
        Obtiene datos para múltiples timeframes (asíncrono)
        """
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

        data = {}
        tasks = []
        
        # Crear tareas para cada timeframe
        for tf in timeframes:
            task = self.obtener_ohlcv_async(symbol, tf, limit)
            tasks.append((tf, task))
        
        # Ejecutar concurrentemente
        for tf, task in tasks:
            try:
                df = await task
                if not df.empty:
                    data[tf] = df
                    self.logger.info(f"✅ Datos async {tf} obtenidos para {symbol}")
                else:
                    self.logger.warning(f"⚠️ Sin datos async {tf} para {symbol}")
            except Exception as e:
                self.logger.error(f"❌ Error en timeframe {tf} para {symbol}: {e}")
                continue

        return data

    def obtener_datos_para_riesgo(self, symbol, periodos_volatilidad=30):
        """
        Obtiene datos específicos para análisis de riesgo
        """
        try:
            # Datos para volatilidad (1h para análisis más estable)
            df_vol = self.obtener_ohlcv(symbol, "1h", limit=periodos_volatilidad * 24)  # ~30 días
            
            # Datos para correlación (1d para análisis de largo plazo)
            df_corr = self.obtener_ohlcv(symbol, "1d", limit=90)  # 90 días
            
            # Datos actuales para precio y volumen
            df_actual = self.obtener_ohlcv(symbol, "5m", limit=100)
            
            return {
                'volatilidad': df_vol,
                'correlacion': df_corr,
                'actual': df_actual
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos para riesgo {symbol}: {e}")
            return {}

    async def obtener_datos_mercado_completo(self, symbols, timeframes=None):
        """
        Obtiene datos completos de mercado para múltiples símbolos
        """
        if timeframes is None:
            timeframes = ['15m', '1h', '4h']

        market_data = {}
        
        for symbol in symbols:
            try:
                symbol_data = await self.obtener_datos_multitimeframe_async(symbol, timeframes)
                if symbol_data:
                    market_data[symbol] = symbol_data
                    self.logger.info(f"✅ Datos completos obtenidos para {symbol}")
                else:
                    self.logger.warning(f"⚠️ Sin datos completos para {symbol}")
            except Exception as e:
                self.logger.error(f"❌ Error obteniendo datos completos para {symbol}: {e}")
                continue
                
            # Pequeña pausa para no saturar la API
            await asyncio.sleep(0.5)

        return market_data

    def _procesar_dataframe(self, df):
        """
        Procesamiento común para DataFrames
        """
        try:
            # Columnas esenciales
            essential_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Verificar columnas
            missing_cols = [col for col in essential_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Columnas faltantes: {missing_cols}")
                return pd.DataFrame()

            # Convertir tipos de datos
            df = df[essential_cols].copy()
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Convertir timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Limpieza
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]
            
            # Verificar que tenemos datos suficientes
            if len(df) < 5:
                return pd.DataFrame()
                
            # Formatear para estrategias
            df = self._formatear_para_estrategias(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error procesando DataFrame: {e}")
            return pd.DataFrame()

    def _formatear_para_estrategias(self, df):
        """
        Formatea el DataFrame para compatibilidad con las nuevas estrategias
        """
        try:
            # Asegurar que tenemos el formato correcto
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Verificar y renombrar si es necesario
            column_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Asegurar que tenemos todas las columnas requeridas
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Columna requerida faltante después del mapeo: {col}")
                    return pd.DataFrame()
            
            # Ordenar por timestamp
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error formateando DataFrame para estrategias: {e}")
            return pd.DataFrame()

    def limpiar_cache(self, older_than_seconds=None):
        """
        Limpia el cache de datos
        """
        try:
            if older_than_seconds is None:
                self.cache.clear()
                self.logger.info("Cache limpiado completamente")
            else:
                current_time = time.time()
                keys_to_remove = [
                    key for key, (_, timestamp) in self.cache.items()
                    if current_time - timestamp > older_than_seconds
                ]
                for key in keys_to_remove:
                    del self.cache[key]
                self.logger.info(f"Cache limpiado: {len(keys_to_remove)} entradas removidas")
                
        except Exception as e:
            self.logger.error(f"Error limpiando cache: {e}")

    def get_cache_info(self):
        """
        Obtiene información del cache
        """
        return {
            'total_entries': len(self.cache),
            'cache_timeout': self.cache_timeout,
            'keys': list(self.cache.keys())
        }

    def obtener_info_mercado(self, symbol):
        """
        Obtiene información adicional del mercado
        """
        try:
            # Información del símbolo
            exchange_info = self.client.get_exchange_info()
            symbol_info = next(
                (s for s in exchange_info['symbols'] if s['symbol'] == symbol), 
                None
            )
            
            # Ticker de 24h
            ticker_24h = self.client.get_24hr_ticker(symbol=symbol)
            
            # Order book (profundidad)
            depth = self.client.get_order_book(symbol=symbol, limit=10)
            
            return {
                'symbol_info': symbol_info,
                'ticker_24h': ticker_24h,
                'order_book': depth
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo info mercado para {symbol}: {e}")
            return {}

# Utilidad para conversión de datos legacy
class DataAdapter:
    @staticmethod
    def adaptar_para_estrategias(df):
        """
        Adapta DataFrames legacy al formato de nuevas estrategias
        """
        if df is None or df.empty:
            return df
            
        # Copiar para no modificar el original
        df_adapted = df.copy()
        
        # Mapeo de columnas
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
            'open': 'open', 'high': 'high', 'low': 'low', 
            'close': 'close', 'volume': 'volume'
        }
        
        # Renombrar columnas
        df_adapted = df_adapted.rename(columns=column_mapping)
        
        # Asegurar índice de timestamp
        if not isinstance(df_adapted.index, pd.DatetimeIndex):
            if 'timestamp' in df_adapted.columns:
                df_adapted['timestamp'] = pd.to_datetime(df_adapted['timestamp'])
                df_adapted.set_index('timestamp', inplace=True)
            elif 'date' in df_adapted.columns:
                df_adapted['date'] = pd.to_datetime(df_adapted['date'])
                df_adapted.set_index('date', inplace=True)
        
        # Ordenar por tiempo
        df_adapted = df_adapted.sort_index()
        
        return df_adapted

    @staticmethod
    def crear_datos_multitimeframe(symbol_data_dict):
        """
        Convierte diccionario de datos a formato multi-timeframe
        """
        multi_tf_data = {}
        
        for timeframe, df in symbol_data_dict.items():
            adapted_df = DataAdapter.adaptar_para_estrategias(df)
            if adapted_df is not None and not adapted_df.empty:
                multi_tf_data[timeframe] = adapted_df
        
        return multi_tf_data