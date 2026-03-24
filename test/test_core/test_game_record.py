from pymcts.core.game_record import MoveRecord, GameRecord, GameRecordCollection, EvalResult


class TestGameRecord:
    def test_move_record_creation(self):
        rec = MoveRecord(action=5, player=0, policy=None)
        assert rec.action == 5
        assert rec.player == 0

    def test_game_record_creation(self):
        moves = [MoveRecord(action=0, player=0, policy=None)]
        record = GameRecord(
            game_type="test",
            game_config={"size": 3},
            moves=moves,
            winner=0,
            player_names=["p0", "p1"],
        )
        assert record.num_moves == 1
        assert record.winner == 0

    def test_collection_evaluate(self):
        moves = [MoveRecord(action=0, player=0, policy=None)]
        records = [
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=0, player_names=["alice", "bob"]),
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=1, player_names=["alice", "bob"]),
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=0, player_names=["alice", "bob"]),
        ]
        collection = GameRecordCollection(game_records=records)
        result = collection.evaluate("alice")
        assert result.wins == 2
        assert result.losses == 1
        assert result.draws == 0
        assert result.win_rate == 2 / 3

    def test_collection_is_better(self):
        moves = [MoveRecord(action=0, player=0, policy=None)]
        records = [
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=0, player_names=["alice", "bob"])
            for _ in range(6)
        ] + [
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=1, player_names=["alice", "bob"])
            for _ in range(4)
        ]
        collection = GameRecordCollection(game_records=records)
        assert collection.is_better("alice", 0.55) is True
        assert collection.is_better("bob", 0.55) is False

    def test_collection_handles_draws(self):
        moves = [MoveRecord(action=0, player=0, policy=None)]
        records = [
            GameRecord(game_type="test", game_config={}, moves=moves,
                       winner=None, player_names=["alice", "bob"]),
        ]
        collection = GameRecordCollection(game_records=records)
        result = collection.evaluate("alice")
        assert result.draws == 1
        assert result.total == 1
        assert result.win_rate == 0.0
