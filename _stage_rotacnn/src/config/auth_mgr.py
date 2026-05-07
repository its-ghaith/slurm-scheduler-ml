import json
import logging
import os
import stat
import time
from enum import Enum
from pathlib import Path

import requests

log = logging.getLogger(__name__)


class TokenType(Enum):
	"""OIDC Token Types"""

	ACCESS_TOKEN = 1, "Access Tokens"
	ID_TOKEN = 2, "JWT ID Token, usefull when using with OAUTH"


class AuthMgr:
	"""Simple OIDC Authentication Manager

	Managing authentication against an GitLab OIDC Server using device grants."""

	def __init__(self, oidc_url, client_id, cache_file: Path | None = None):
		"""Initalise the Authentication Manager

		Keyword arguments:
		oidc_url -- url to the OIDC server
		client_id -- Client ID for the tokenretrval
		"""

		self.client_id = client_id
		self.oidc_url = oidc_url
		self.id_token = None
		self.access_token = None
		self.refresh_token = None
		self.expires = 0
		self.cache_file = cache_file

	def login(self, poll_intervall_s=1):
		"""Perform the Serverlogin

		Keyword arguments:
		poll_intervall_s -- inital polling period for device authorisaton. Is increased if requested by the server.
		"""
		r = requests.post(
			self.oidc_url + "authorize_device", params={"client_id": self.client_id, "scope": "openid profile email"}
		)
		if not r.ok:
			print(r.text)
			data = json.loads(r.text)
			raise RuntimeError(f"{data['error']}: {data['error_description']}")
		data = json.loads(r.text)
		device_code = data["device_code"]
		print(f"Please log into gitlab: {data['verification_uri_complete']}")

		while True:
			time.sleep(poll_intervall_s)
			r = requests.post(
				self.oidc_url + "token",
				params={
					"grant_type": "urn:ietf:params:oauth:grant-type:device_code",
					"client_id": self.client_id,
					"device_code": device_code,
				},
			)
			data = json.loads(r.text)
			if r.ok:
				break
			elif data["error"] == "authorization_pending":
				continue
			elif data["error"] == "slow_down":
				poll_intervall_s = poll_intervall_s * 2
				log.info(f"Polling to fast, reducing to {poll_intervall_s}")
			else:
				raise RuntimeError(f"{data['error']}: {data['error_description']}")

		print("Authentication Successful")

		# log.info(f"jwt:{jwt.decode(data['id_token'], options={'verify_signature': False})}")

		self.id_token = data["id_token"]
		self.access_token = data["access_token"]
		self.refresh_token = data["refresh_token"]
		self.expires = data["created_at"] + data["expires_in"]

	def write_cache(self):
		"""Writes the refresh_token to the given cache_file path.
		WARNIG: the refresh token can be userd once to generate new access and refresh token. Who ever that token has,
		is able to access OIDC Server on behalv of the user.
		Therefore, the file mode is changed to read for the user only (similar to ssh keys)

		cache_file -- path to the cachefile, (default ./oidc_cache)
		"""
		assert self.cache_file
		self.cache_file.parent.mkdir(parents=True, exist_ok=True)
		with open(self.cache_file, "w") as f:
			json.dump({"refresh_token": self.refresh_token}, f)
		os.chmod(str(self.cache_file), stat.S_IRUSR | stat.S_IWUSR)

	def read_cache(self):
		"""loads refresh token from a cachefile written by save_cache.
		After a call to read_cache, `renew_token`or `get_token` can be used without calling login,
		if the given refreshtoken is valid.

		cache_file -- path to the cache file (default ./oidc_cache)
		"""
		assert self.cache_file
		with open(self.cache_file) as f:
			d = json.load(f)
			self.expires = 1
			self.refresh_token = d["refresh_token"]

	def renew_token(self):
		"""renew tokens if they are expired"""
		r = requests.post(
			self.oidc_url + "token",
			params={"grant_type": "refresh_token", "client_id": self.client_id, "refresh_token": self.refresh_token},
		)
		if not r.ok:
			raise RuntimeError(r.text)
		data = json.loads(r.text)
		# log.info(f"jwt:{jwt.decode(data['id_token'], options={'verify_signature': False})}")
		self.id_token = data["id_token"]
		self.access_token = data["access_token"]
		self.refresh_token = data["refresh_token"]
		self.write_cache()

	def get_token(self, token_type=TokenType.ID_TOKEN):
		"""Get token

		Keyword arguments:
		token_type -- one of TokenType (default TokenType.ID_TOKEN)
		"""
		if self.expires == 0:
			raise RuntimeError("Login required before a token can be aquired.")
		self.renew_token()

		if token_type == TokenType.ID_TOKEN:
			return self.id_token
		elif token_type == TokenType.ACCESS_TOKEN:
			return self.access_token
		else:
			raise RuntimeError("Unkown token type")
