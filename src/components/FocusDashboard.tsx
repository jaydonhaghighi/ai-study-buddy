import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { collection, onSnapshot, query, Timestamp, where } from 'firebase/firestore';
import { db } from '../firebase-config';
import './FocusDashboard.css';

const COLORS: Record<string, string> = {
  screen: '#10B981', // emerald
  away_left: '#F87171', // red
  away_right: '#FB923C', // orange
  away_up: '#A78BFA', // purple
  away_down: '#60A5FA', // blue
  away_unknown: '#9CA3AF', // gray
};

type FocusSummaryDoc = {
  id: string;
  userId: string;
  focusSessionId?: string;
  courseId?: string | null;
  sessionId?: string | null; // preferred (matches focusSessions.sessionId)
  courseSessionId?: string | null; // legacy/alternate
  createdAt?: Date | null;
  startTs?: number | null; // epoch seconds
  endTs?: number | null; // epoch seconds
  focusedMs?: number | null;
  distractedMs?: number | null;
  distractions?: number | null;
  focusPercent?: number | null;
  attentionLabelCounts?: Record<string, number> | null;
};

type Course = { id: string; name: string; userId: string };
type Session = { id: string; name: string; userId: string; courseId: string };

function toDateMaybe(v: unknown): Date | null {
  if (!v) return null;
  if (v instanceof Date) return v;
  if (v instanceof Timestamp) return v.toDate();
  return null;
}

function formatShortDate(d: Date) {
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function formatDateTime(d: Date) {
  return d.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function msToMinutes(ms: number | null | undefined): number {
  if (ms == null) return 0;
  return Math.max(0, ms) / 60000;
}

function formatHMS(ms: number | null | undefined, zeroPlaceholder: string = '—'): string {
  const totalSeconds = Math.max(0, Math.floor((ms ?? 0) / 1000));
  if (totalSeconds === 0) return zeroPlaceholder;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function fmtAwayLabel(k: string): string {
  switch (k) {
    case 'away_left': return 'Left';
    case 'away_right': return 'Right';
    case 'away_up': return 'Up';
    case 'away_down': return 'Down';
    default:
      return k.replace(/^away_/, '').replace(/_/g, ' ');
  }
}

function focusTier(pct: number | null | undefined): 'high' | 'mid' | 'low' | 'na' {
  if (pct == null || !Number.isFinite(pct)) return 'na';
  if (pct >= 70) return 'high';
  if (pct >= 40) return 'mid';
  return 'low';
}

export default function FocusDashboard({ userId }: { userId: string }) {
  const [summaries, setSummaries] = useState<FocusSummaryDoc[]>([]);
  const [summariesError, setSummariesError] = useState<string | null>(null);
  const [courses, setCourses] = useState<Course[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<string>('__all__');

  useEffect(() => {
    const q = query(
      collection(db, 'focusSummaries'),
      where('userId', '==', userId)
    );

    setSummariesError(null);
    return onSnapshot(
      q,
      (snap) => {
        const rows: FocusSummaryDoc[] = snap.docs.map((d) => {
          const data = d.data() as any;
          return {
            id: d.id,
            userId: data.userId,
            focusSessionId: data.focusSessionId,
            courseId: data.courseId ?? null,
            sessionId: data.sessionId ?? null,
            courseSessionId: data.courseSessionId ?? null,
            createdAt: toDateMaybe(data.createdAt),
            startTs: typeof data.startTs === 'number' ? data.startTs : null,
            endTs: typeof data.endTs === 'number' ? data.endTs : null,
            focusedMs: typeof data.focusedMs === 'number' ? data.focusedMs : null,
            distractedMs: typeof data.distractedMs === 'number' ? data.distractedMs : null,
            distractions: typeof data.distractions === 'number' ? data.distractions : 0,
            focusPercent: typeof data.focusPercent === 'number' ? data.focusPercent : null,
            attentionLabelCounts: (data.attentionLabelCounts && typeof data.attentionLabelCounts === 'object') ? data.attentionLabelCounts : null,
          };
        });

        // Sort newest-first (avoid needing an orderBy index).
        rows.sort((a, b) => {
          const ta = (a.endTs ? a.endTs * 1000 : (a.createdAt ? a.createdAt.getTime() : 0));
          const tb = (b.endTs ? b.endTs * 1000 : (b.createdAt ? b.createdAt.getTime() : 0));
          return tb - ta;
        });

        setSummaries(rows);
        setSummariesError(null);
      },
      (err) => {
        setSummaries([]);
        setSummariesError(err?.message || 'Failed to load focus summaries');
      }
    );
  }, [userId]);

  useEffect(() => {
    const q = query(collection(db, 'courses'), where('userId', '==', userId));
    return onSnapshot(q, (snap) => {
      setCourses(snap.docs.map((d) => ({ id: d.id, ...(d.data() as any) } as Course)));
    });
  }, [userId]);

  useEffect(() => {
    const q = query(collection(db, 'sessions'), where('userId', '==', userId));
    return onSnapshot(q, (snap) => {
      setSessions(snap.docs.map((d) => ({ id: d.id, ...(d.data() as any) } as Session)));
    });
  }, [userId]);

  const courseNameById = useMemo(() => {
    const m = new Map<string, string>();
    for (const c of courses) m.set(c.id, c.name);
    return m;
  }, [courses]);

  const sessionNameById = useMemo(() => {
    const m = new Map<string, string>();
    for (const s of sessions) m.set(s.id, s.name);
    return m;
  }, [sessions]);

  const filtered = useMemo(() => {
    if (selectedCourseId === '__all__') return summaries;
    return summaries.filter((s) => s.courseId === selectedCourseId);
  }, [summaries, selectedCourseId]);

  const series = useMemo(() => {
    // Render oldest -> newest for charts
    const ordered = [...filtered].reverse();
    return ordered.map((s, idx) => {
      const end = s.endTs ? new Date(s.endTs * 1000) : (s.createdAt ?? null);
      const label = end ? formatShortDate(end) : `#${idx + 1}`;
      return {
        key: s.id,
        label,
        when: end ? end.getTime() : idx,
        focusPercent: s.focusPercent ?? 0,
        focusedMin: Number(msToMinutes(s.focusedMs).toFixed(1)),
        distractedMin: Number(msToMinutes(s.distractedMs).toFixed(1)),
        distractions: s.distractions ?? 0,
      };
    });
  }, [filtered]);

  const latestRows = useMemo(() => filtered.slice(0, 25), [filtered]);
  const latest = filtered[0] || null;

  const topAwayLabel = (counts: Record<string, number> | null | undefined): string | null => {
    if (!counts) return null;
    const entries = Object.entries(counts)
      .filter(([k]) => k !== 'screen')
      .sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) return null;
    return entries[0][0];
  };

  const lastSessionDurationMin = useMemo(() => {
    if (!latest) return 0;
    const totalMs = (latest.focusedMs ?? 0) + (latest.distractedMs ?? 0);
    return totalMs;
  }, [latest]);

  const lastTopAwayPretty = useMemo(() => {
    const k = topAwayLabel(latest?.attentionLabelCounts);
    return k ? fmtAwayLabel(k) : '—';
  }, [latest]);

  const lastAttentionPie = useMemo(() => {
    const counts = latest?.attentionLabelCounts || null;
    if (!counts) return [];
    const total = Object.values(counts).reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
    if (total <= 0) return [];
    const order = ['screen', 'away_left', 'away_right', 'away_up', 'away_down'];
    return order
      .filter((k) => (counts as any)[k] != null)
      .map((k) => ({
        key: k,
        name: k === 'screen' ? 'Screen' : fmtAwayLabel(k),
        value: (counts as any)[k] as number,
        pct: Math.round((((counts as any)[k] as number) / total) * 100),
        color: COLORS[k] ?? '#111827',
      }))
      .filter((x) => x.value > 0);
  }, [latest]);

  const totals = useMemo(() => {
    const focusedMs = filtered.reduce((acc, s) => acc + (s.focusedMs ?? 0), 0);
    const distractedMs = filtered.reduce((acc, s) => acc + (s.distractedMs ?? 0), 0);
    const avgFocusPercent =
      filtered.length > 0
        ? Number(
            (
              filtered.reduce((acc, s) => acc + (s.focusPercent ?? 0), 0) / filtered.length
            ).toFixed(1)
          )
        : 0;
    return { focusedMs, distractedMs, avgFocusPercent };
  }, [filtered]);

  return (
    <div className="focusdash">
      <div className="focusdash-header">
        <div className="focusdash-controls">
          <label className="focusdash-label">
            Course
            <select
              className="focusdash-select"
              value={selectedCourseId}
              onChange={(e) => setSelectedCourseId(e.target.value)}
            >
              <option value="__all__">All courses</option>
              {courses
                .slice()
                .sort((a, b) => a.name.localeCompare(b.name))
                .map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
            </select>
          </label>
        </div>
      </div>

      <div className="focusdash-metrics">
        <div className="focusdash-metric focusdash-metric--neutral">
          <div className="focusdash-metric-label">Sessions</div>
          <div className="focusdash-metric-value">{filtered.length}</div>
        </div>
        <div className="focusdash-metric focusdash-metric--focus">
          <div className="focusdash-metric-label">Avg focus</div>
          <div className="focusdash-metric-value">{totals.avgFocusPercent}%</div>
        </div>
        <div className="focusdash-metric focusdash-metric--focus">
          <div className="focusdash-metric-label">Focused</div>
          <div className="focusdash-metric-value">{formatHMS(totals.focusedMs)}</div>
        </div>
        <div className="focusdash-metric focusdash-metric--neutral">
          <div className="focusdash-metric-label">Last session</div>
          <div className="focusdash-metric-value">{lastSessionDurationMin ? formatHMS(lastSessionDurationMin) : '—'}</div>
          <div className="focusdash-metric-sub">Top away: {lastTopAwayPretty}</div>
        </div>
      </div>

      {summariesError ? (
        <div className="focusdash-empty">
          <div className="focusdash-empty-title">Couldn’t load focus summaries</div>
          <div className="focusdash-empty-subtitle">{summariesError}</div>
        </div>
      ) : filtered.length === 0 ? (
        <div className="focusdash-empty">
          <div className="focusdash-empty-title">No focus summaries yet</div>
          <div className="focusdash-empty-subtitle">
            Start and stop a focus session with webcam tracking enabled, then come back here.
          </div>
        </div>
      ) : (
        <>
          <div className="focusdash-grid">
            <div className="focusdash-card">
              <div className="focusdash-card-title">Focus % over time</div>
              <div className="focusdash-chart">
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={series}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis
                      dataKey="when"
                      type="number"
                      domain={['dataMin', 'dataMax']}
                      tick={{ fontSize: 12, fill: '#111827' }}
                      tickFormatter={(v) => {
                        const n = typeof v === 'number' ? v : Number(v);
                        if (!Number.isFinite(n)) return '';
                        return formatShortDate(new Date(n));
                      }}
                    />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#111827' }} />
                    <Tooltip
                      labelFormatter={(v) => {
                        const n = typeof v === 'number' ? v : Number(v);
                        if (!Number.isFinite(n)) return '';
                        return formatDateTime(new Date(n));
                      }}
                      formatter={(value: any) => {
                        const n = typeof value === 'number' ? value : Number(value);
                        if (!Number.isFinite(n)) return ['—', 'Focus %'];
                        return [`${Math.round(n)}%`, 'Focus %'];
                      }}
                      contentStyle={{
                        background: '#FFFFFF',
                        border: '1px solid rgba(17, 24, 39, 0.14)',
                        borderRadius: 10,
                        color: '#111827',
                      }}
                      labelStyle={{ color: '#111827', fontWeight: 700 }}
                    />
                    <Line type="monotone" dataKey="focusPercent" stroke="#111827" strokeWidth={2.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="focusdash-card">
              <div className="focusdash-card-title">Last session attention breakdown</div>
              <div className="focusdash-split">
                <div className="focusdash-split-left">
                  {lastAttentionPie.length === 0 ? (
                    <div className="focusdash-card-hint">No attention direction data yet.</div>
                  ) : (
                    <ResponsiveContainer width="100%" height={220}>
                      <PieChart>
                        <Pie
                          data={lastAttentionPie}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          innerRadius={42}
                          stroke="#111827"
                          strokeWidth={1}
                        >
                          {lastAttentionPie.map((entry) => (
                            <Cell
                              key={entry.key}
                              fill={(entry as any).color ?? '#111827'}
                              stroke="#111827"
                              strokeWidth={1}
                            />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(value: any, _name: any, props: any) => {
                            const pct = props?.payload?.pct;
                            const dir = props?.payload?.name ?? 'Direction';
                            return [`${value} frames (${pct ?? 0}%)`, dir];
                          }}
                          contentStyle={{
                            background: '#FFFFFF',
                            border: '1px solid #111827',
                            borderRadius: 10,
                            color: '#111827',
                          }}
                          labelStyle={{ color: '#111827', fontWeight: 700 }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  )}
                </div>
                <div className="focusdash-split-right">
                  <div className="focusdash-stat">
                    <div className="focusdash-stat-label">Top away direction</div>
                    <div className="focusdash-stat-value">{lastTopAwayPretty}</div>
                  </div>
                  <div className="focusdash-stat">
                    <div className="focusdash-stat-label">Focus %</div>
                    <div className="focusdash-stat-value">
                      {latest?.focusPercent == null ? '—' : (
                        <span className={`focusdash-badge focusdash-badge--${focusTier(latest.focusPercent)}`}>
                          {Math.round(latest.focusPercent)}%
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="focusdash-stat">
                    <div className="focusdash-stat-label">Distractions</div>
                    <div className="focusdash-stat-value">{latest?.distractions ?? 0}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="focusdash-card focusdash-tablecard">
            <div className="focusdash-card-title">Recent sessions (clean view)</div>
            <div className="focusdash-tablewrap">
              <table className="focusdash-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Course</th>
                    <th>Session</th>
                    <th>Duration</th>
                    <th>Focused</th>
                    <th>Distracted</th>
                    <th>Focus %</th>
                    <th>Distractions</th>
                    <th>Top away</th>
                  </tr>
                </thead>
                <tbody>
                  {latestRows.map((s) => {
                    const end = s.endTs ? new Date(s.endTs * 1000) : (s.createdAt ?? null);
                    const courseName = s.courseId ? (courseNameById.get(s.courseId) ?? s.courseId.slice(0, 8) + '…') : '—';
                    const linkedSessionId = s.sessionId ?? s.courseSessionId ?? null;
                    const sessionName = linkedSessionId ? (sessionNameById.get(linkedSessionId) ?? linkedSessionId.slice(0, 8) + '…') : '—';
                    const durationMs = (s.focusedMs ?? 0) + (s.distractedMs ?? 0);
                    const focusedMs = s.focusedMs ?? 0;
                    const distractedMs = s.distractedMs ?? 0;
                    const focusPct = s.focusPercent == null ? null : Math.round(s.focusPercent);
                    const topAwayRaw = topAwayLabel(s.attentionLabelCounts);
                    const topAway = topAwayRaw ? fmtAwayLabel(topAwayRaw) : '—';
                    return (
                      <tr key={s.id}>
                        <td>{end ? formatDateTime(end) : '—'}</td>
                        <td>{courseName}</td>
                        <td>{sessionName}</td>
                        <td>{formatHMS(durationMs)}</td>
                        <td>{formatHMS(focusedMs)}</td>
                        <td>{formatHMS(distractedMs)}</td>
                        <td>
                          {focusPct == null ? '—' : (
                            <span className={`focusdash-badge focusdash-badge--${focusTier(focusPct)}`}>
                              {focusPct}%
                            </span>
                          )}
                        </td>
                        <td>{s.distractions ?? 0}</td>
                        <td>
                          <span
                            className="focusdash-away"
                            style={{
                              borderColor: COLORS[topAwayRaw ?? 'away_unknown'] ?? '#D1D5DB',
                              background: topAwayRaw ? `${(COLORS[topAwayRaw] ?? '#9CA3AF')}22` : 'transparent',
                            }}
                          >
                            {topAway}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

