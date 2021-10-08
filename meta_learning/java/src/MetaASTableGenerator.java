import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.SQLException;
import java.sql.Time;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.basic.StatisticsUtil;
import ai.libs.jaicore.basic.ValueUtil;
import ai.libs.jaicore.basic.kvstore.KVStore;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreCollectionOneLayerPartition;
import ai.libs.jaicore.basic.kvstore.KVStoreCollectionTwoLayerPartition;
import ai.libs.jaicore.basic.kvstore.KVStoreSequentialComparator;
import ai.libs.jaicore.basic.kvstore.KVStoreStatisticsUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreUtil;
import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;

public class MetaASTableGenerator {

	private static final String DB_CONFIG = "db.properties";
	private static final boolean LOAD_FROM_DB = false;
	private static final boolean OVERWRITE_DUMP = false;
	private static final String DB_DUMP_FILENAME = "table-data.kvcol";
	private static final String DB_DUMP2_FILENAME = "table2-data.kvcol";

	private static final String TBL_1 = "normalized_par10_level_0";
	private static final String TBL_2 = "normalized_par10_level_1";
	private static final String TBL_3 = "normalized_by_level_0_par10_level_1";

	private static final int TRIM_ELEMENTS = 2;

	private static KVStoreCollection col1;
	private static KVStoreCollection col2;

	private static final String OUT_WTL = "win-tie-loss-stats.tex";
	private static final String OUT_BASE_TABLE = "base-table.tex";
	private static final String OUT_META_TABLE = "meta-table.tex";

	private static void loadData() throws SQLException {
		if (LOAD_FROM_DB) {
			IDatabaseConfig config = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File(DB_CONFIG));
			IDatabaseAdapter adapter = DatabaseAdapterFactory.get(config);
			col1 = KVStoreUtil.readFromMySQLQuery(adapter,
					"SELECT * FROM ((SELECT * FROM " + TBL_3 + " UNION SELECT * FROM " + TBL_1 + ") as union_table) WHERE approach NOT IN ('sbs', 'oracle', 'sbs_with_feature_costs','l1_sbs', 'l1_oracle', 'l1_sbs_with_feature_costs')",
					new HashMap<>());
			col1.removeAny(new String[] { "sbs", "oracle" }, true);
			col1 = col1.group("scenario_name", "approach");
			col1.setCollectionID("DB-Dump-" + new Time(System.currentTimeMillis()));

			// col2 = KVStoreUtil.readFromMySQLTable(adapter, TBL_2, new HashMap<>());
			col2 = KVStoreUtil.readFromMySQLQuery(adapter,
					"SELECT oracle_and_sbs_table.scenario_name, oracle_and_sbs_table.fold, server_results_meta_level_1.approach, oracle_and_sbs_table.metric, server_results_meta_level_1.result, ((server_results_meta_level_1.result - oracle_and_sbs_table.oracle_result)/(oracle_and_sbs_table.sbs_result -oracle_and_sbs_table.oracle_result)) as n_par10,oracle_and_sbs_table.oracle_result, oracle_and_sbs_table.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `server_results_meta_level_1` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `server_results_meta_level_1` WHERE approach='sbs_with_feature_costs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as oracle_and_sbs_table JOIN server_results_meta_level_1 ON oracle_and_sbs_table.scenario_name = server_results_meta_level_1.scenario_name AND oracle_and_sbs_table.fold = server_results_meta_level_1.fold AND oracle_and_sbs_table.metric = server_results_meta_level_1.metric WHERE oracle_and_sbs_table.metric='par10'",
					new HashMap<>());
			col2.removeAny(new String[] { "approach=sbs", "approach=oracle" }, true);
			col2 = col2.group("scenario_name", "approach");
			col2.setCollectionID("DB-Dump2-" + new Time(System.currentTimeMillis()));

			if (OVERWRITE_DUMP) {
				try {
					col1.serializeTo(new File(DB_DUMP_FILENAME));
					col2.serializeTo(new File(DB_DUMP2_FILENAME));
				} catch (IOException e1) {
					e1.printStackTrace();
				}
			}
		} else {
			try {
				col1 = new KVStoreCollection(FileUtil.readFileAsString(new File(DB_DUMP_FILENAME)));
				col2 = new KVStoreCollection(FileUtil.readFileAsString(new File(DB_DUMP2_FILENAME)));
			} catch (IOException e) {
				e.printStackTrace();
				System.err.println("Could not load local dump. Exiting.");
			}
		}
	}

	public static void main(final String[] args) throws SQLException {
		loadData();

		Map<String, String> replacement = new HashMap<>();
		replacement.put("Expectation_algorithm_survival_forest", "00-R2SExp");
		replacement.put("PAR10_algorithm_survival_forest", "10-R2SPAR10");
		replacement.put("isac", "20-ISAC");
		replacement.put("multiclass_algorithm_selector", "30-MLC");
		replacement.put("per_algorithm_RandomForestRegressor_regressor", "35-PAReg");
		replacement.put("satzilla-11", "40-Satzilla");
		replacement.put("sunny", "50-Sunny");

		col1.stream().forEach(x -> {
			String approach = x.getAsString("approach");

			if (approach.startsWith("l1")) {
				x.put("approach", "l1-" + replacement.computeIfAbsent(approach.substring(3), t -> t));
			} else {
				x.put("approach", replacement.computeIfAbsent(approach, t -> t));
			}
		});

		for (IKVStore store : col1) {
			List<Double> list = store.getAsDoubleList("n_par10");
			Collections.sort(list);
			for (int i = 0; i < TRIM_ELEMENTS; i++) {
				list.remove(0);
				list.remove(list.size() - 1);
			}
			store.put("tm_n_par10", ValueUtil.valueToString(StatisticsUtil.mean(list), 2));
		}

		KVStoreStatisticsUtil.rank(col1, "scenario_name", "approach", "tm_n_par10", "rank");

		// Win Tie Loss Statistics
		KVStoreCollection baseApproaches = new KVStoreCollection();
		col1.stream().filter(x -> !x.getAsString("approach").startsWith("l1")).forEach(baseApproaches::add);

		KVStoreCollectionTwoLayerPartition scenarioWiseBaseApproaches = new KVStoreCollectionTwoLayerPartition("approach", "scenario_name", baseApproaches);

		KVStoreCollection metaApproaches = new KVStoreCollection();
		col1.stream().filter(x -> x.getAsString("approach").startsWith("l1")).forEach(metaApproaches::add);

		KVStoreCollectionTwoLayerPartition partition = new KVStoreCollectionTwoLayerPartition("approach", "scenario_name", metaApproaches);

		// compute statistics
		KVStoreCollectionTwoLayerPartition metaPart = new KVStoreCollectionTwoLayerPartition("approach", "scenario_name", metaApproaches);
		KVStoreCollectionOneLayerPartition basePart = new KVStoreCollectionOneLayerPartition("scenario_name", baseApproaches);
		for (Entry<String, Map<String, KVStoreCollection>> metaPartEntry : metaPart) {
			for (Entry<String, KVStoreCollection> metaPartScenarioEntry : metaPartEntry.getValue().entrySet()) {
				if (metaPartScenarioEntry.getValue().size() > 1) {
					throw new IllegalStateException("meta part scenario entry has more than 1 store");
				}

				int win = 0;
				int tie = 0;
				int loss = 0;
				IKVStore metaStore = metaPartScenarioEntry.getValue().get(0);
				for (IKVStore baseStore : basePart.getData().get(metaPartScenarioEntry.getKey())) {
					switch (baseStore.getAsDouble("tm_n_par10").compareTo(metaStore.getAsDouble("tm_n_par10"))) {
					case 1:
						win++;
						break;
					case 0:
						tie++;
						break;
					case -1:
						loss++;
						break;
					}
				}

				metaStore.put("statistics", (win + tie) + "/" + loss);
			}
		}

		KVStoreCollection countData = new KVStoreCollection();
		for (Entry<String, Map<String, KVStoreCollection>> metaPartEntry : partition) {
			for (Entry<String, Map<String, KVStoreCollection>> basePartEntry : scenarioWiseBaseApproaches) {
				int win = 0;
				int tie = 0;
				int loss = 0;

				for (String key : metaPartEntry.getValue().keySet()) {
					IKVStore metaStore = metaPartEntry.getValue().get(key).get(0);
					IKVStore baseStore = basePartEntry.getValue().get(key).get(0);
					switch (metaStore.getAsDouble("tm_n_par10").compareTo(baseStore.getAsDouble("tm_n_par10"))) {
					case 1:
						win++;
						break;
					case 0:
						tie++;
						break;
					case -1:
						loss++;
						break;
					}
				}

				IKVStore store = new KVStore();
				store.put("meta_approach", metaPartEntry.getKey());
				store.put("base_approach", basePartEntry.getKey());

				store.put("win", win);
				store.put("tie", tie);
				store.put("loss", loss);

				store.put("entry", win + "/" + tie + "/" + loss);
				countData.add(store);
			}
		}

		countData.sort(new KVStoreSequentialComparator("scenario_name", "approach"));
		writeFile(KVStoreUtil.kvStoreCollectionToLaTeXTable(countData, "base_approach", "meta_approach", "entry"), OUT_WTL);

		KVStoreCollectionOneLayerPartition meanWTLStats = new KVStoreCollectionOneLayerPartition("approach", countData);
		for (Entry<String, KVStoreCollection> entry : meanWTLStats) {
			List<Double> win = new ArrayList<>();
			List<Double> tie = new ArrayList<>();
			List<Double> loss = new ArrayList<>();

			for (IKVStore store : entry.getValue()) {
				win.add(store.getAsDouble("win"));
				tie.add(store.getAsDouble("tie"));
				loss.add(store.getAsDouble("loss"));
			}
		}

		// Result Table (cont.)
		KVStoreStatisticsUtil.best(col1, "scenario_name", "approach", "tm_n_par10", "best");
		col1.stream().forEach(x -> {
			if (x.getAsBoolean("best")) {
				x.put("tm_n_par10", "\\textbf{" + x.getAsString("tm_n_par10") + "}");
			}

			if (x.getAsString("approach").startsWith("l1")) {
				x.put("tm_n_par10", x.getAsString("tm_n_par10") + " (" + x.getAsString("statistics") + ")");
			}
		});

		col1.sort(new KVStoreSequentialComparator("scenario_name", "approach"));

		writeFile(KVStoreUtil.kvStoreCollectionToLaTeXTable(col1, "scenario_name", "approach", "tm_n_par10"), OUT_BASE_TABLE);

		col2.stream().forEach(x -> {
			String approach = x.getAsString("approach");

			if (approach.startsWith("l1")) {
				x.put("approach", "l1_" + replacement.computeIfAbsent(approach.substring(3), t -> t));
			} else {
				x.put("approach", replacement.computeIfAbsent(approach, t -> t));
			}
		});

		for (IKVStore store : col2) {
			List<Double> list = store.getAsDoubleList("n_par10");
			Collections.sort(list);
			for (int i = 0; i < TRIM_ELEMENTS; i++) {
				list.remove(0);
				list.remove(list.size() - 1);
			}
			store.put("tm_n_par10", ValueUtil.valueToString(StatisticsUtil.mean(list), 2));
		}

		KVStoreStatisticsUtil.best(col2, "scenario_name", "approach", "tm_n_par10", "best");
		col2.stream().forEach(x -> {
			if (x.getAsDouble("tm_n_par10") < 1) {
				x.put("tm_n_par10", "\\textbf{" + x.getAsString("tm_n_par10") + "}");
			}
		});

		col2.sort(new KVStoreSequentialComparator("scenario_name", "approach"));

		writeFile(KVStoreUtil.kvStoreCollectionToLaTeXTable(col2, "scenario_name", "approach", "tm_n_par10"), OUT_META_TABLE);
	}

	public static double median(final Collection<? extends Number> values) {
		List<? extends Number> list = new ArrayList<>(values);
		list.sort(new Comparator<Number>() {
			@Override
			public int compare(final Number o1, final Number o2) {
				return Double.compare(o1.doubleValue(), o2.doubleValue());
			}
		});
		int upperIndex = (int) Math.ceil(((double) values.size() + 1) / 2);
		int lowerIndex = (int) Math.floor(((double) values.size() + 1) / 2);

		return (list.get(lowerIndex).doubleValue() + list.get(upperIndex).doubleValue()) / 2;
	}

	private static final void writeFile(final String content, final String filename) {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File(filename)))) {
			bw.write(content);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
