from typing import List, Iterator
from pathlib import Path
from copy import deepcopy
import json
import pandas as pd
import logging


class FmaDatasetReader:
    def __init__(self,
                 config: dict):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.dataset_config = deepcopy(config['datasets']['fma'])
        self.data_path = Path(self.dataset_config['data_path']).expanduser()
        self.genres = self._read_genres()
        self.logger.info('A list of genres:')
        for genre in self.genres:
            self.logger.info(f'{genre["genre_id"]:5}: {genre["title"]}')
        self.logger.info('\n')

        self.tracks = self._read_tracks()
        self.logger.info(f'There is collected {len(self.tracks)} tracks.')

    def _read_genres(self) -> List[dict]:
        genre_subset = [str(genre).strip().lower()
                        for genre in list(self.dataset_config.get('genres', []))]
        meta_path = Path(self.dataset_config['meta_path']).expanduser()
        genres_csv = pd.read_csv(meta_path / 'genres.csv', index_col=0)
        
        all_genre_set = set()
        genres = []
        for genre_id, title in enumerate(genres_csv['title']):
            genre_id += 1
            title = title.strip().lower()
            genres.append({'genre_id': genre_id, 'title': title})
            all_genre_set.add(title)

        if genre_subset:
            for genre_title in genre_subset:
                if genre_title not in all_genre_set:
                    self.logger.warning(f'The genre "{genre_title}" is not in a common set of genres.')
            genres = [genre for genre in genres if genre['title'] in genre_subset]
        out_counter = 0
        for genre in genres:
            genre['out_id'] = out_counter
            out_counter += 1
        return genres

    def _iter_all_mp3_files(self, data_path: Path) -> Iterator[Path]:
        for file_or_dir in data_path.iterdir():
            if file_or_dir.is_dir():
                for mp3_file in self._iter_all_mp3_files(file_or_dir):
                    yield mp3_file
            else:
                if file_or_dir.suffix == '.mp3':
                    yield file_or_dir

    def _read_tracks(self) -> List[dict]:
        genre_id_2_genre = {}
        for genre in self.genres:
            genre_id_2_genre[genre['genre_id']] = genre

        meta_path = Path(self.dataset_config['meta_path']).expanduser()
        tracks_csv = pd.read_csv(meta_path / 'tracks.csv', index_col=0, header=[0, 1])

        track_id_2_info = {}
        for track_id, track in tracks_csv['track'].iterrows():
            raw_track_genre_list = json.loads(track['genres'])
            track_genre_list = []
            for genre_id in raw_track_genre_list:
                if genre_id in genre_id_2_genre:
                    track_genre_list.append(genre_id_2_genre[genre_id])
            if len(track_genre_list) == 0:
                continue
            track_id_2_info[track_id] = {
                'title': track['title'],
                'genres': track_genre_list
            }

        tracks = []
        for mp3_file in self._iter_all_mp3_files(self.data_path):
            try:
                track_id = int(mp3_file.stem)
            except ValueError:
                self.logger.error(f'Unexpected mp3 filename: {mp3_file}')
                continue
            if track_id not in track_id_2_info:
                continue
            track_info = track_id_2_info[track_id]
            tracks.append({'track_id': track_id,
                           'filepath': mp3_file.absolute(),
                           'title': track_info['title'],
                           'genres': track_info['genres']})
        return tracks

    def __len__(self) -> int:
        return len(self.tracks)

    def get_track(self, index: int) -> dict:
        return self.tracks[index]

    def __iter__(self) -> Iterator[dict]:
        return (track for track in self.tracks)

    def slice_tracks(self, size: int):
        self.tracks = self.tracks[:size]
