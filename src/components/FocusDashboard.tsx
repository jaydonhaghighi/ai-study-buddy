import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  Legend,
} from 'recharts';
import { collection, onSnapshot, query, Timestamp, where } from 'firebase/firestore';
import { db } from '../firebase-config';
import './FocusDashboard.css';

type FocusSummaryDoc = {
  id: string;
  userId: string;
  focusSessionId?: string;
  deviceId?: string | null;
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
            deviceId: data.deviceId ?? null,
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
        <div className="focusdash-metric">
          <div className="focusdash-metric-label">Sessions</div>
          <div className="focusdash-metric-value">{filtered.length}</div>
        </div>
        <div className="focusdash-metric">
          <div className="focusdash-metric-label">Avg focus</div>
          <div className="focusdash-metric-value">{totals.avgFocusPercent}%</div>
        </div>
        <div className="focusdash-metric">
          <div className="focusdash-metric-label">Focused (min)</div>
          <div className="focusdash-metric-value">{Math.round(msToMinutes(totals.focusedMs))}</div>
        </div>
        <div className="focusdash-metric">
          <div className="focusdash-metric-label">Distracted (min)</div>
          <div className="focusdash-metric-value">{Math.round(msToMinutes(totals.distractedMs))}</div>
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
            Start and stop a focus session while the Pi agent is running, then come back here.
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
                    <CartesianGrid strokeDasharray="3 3" stroke="#D1D5DB" />
                    <XAxis dataKey="label" tick={{ fontSize: 12, fill: '#111827' }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#111827' }} />
                    <Tooltip
                      contentStyle={{
                        background: '#FFFFFF',
                        border: '1px solid #111827',
                        borderRadius: 10,
                        color: '#111827',
                      }}
                      labelStyle={{ color: '#111827', fontWeight: 700 }}
                    />
                    <Line type="monotone" dataKey="focusPercent" stroke="#111827" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="focusdash-card">
              <div className="focusdash-card-title">Focused vs distracted (minutes)</div>
              <div className="focusdash-chart">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={series.slice(Math.max(0, series.length - 12))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#D1D5DB" />
                    <XAxis dataKey="label" tick={{ fontSize: 12, fill: '#111827' }} />
                    <YAxis tick={{ fontSize: 12, fill: '#111827' }} />
                    <Tooltip
                      contentStyle={{
                        background: '#FFFFFF',
                        border: '1px solid #111827',
                        borderRadius: 10,
                        color: '#111827',
                      }}
                      labelStyle={{ color: '#111827', fontWeight: 700 }}
                    />
                    {/* Custom Legend: both labels have the same sharp, black look and box border */}
                    <Legend
                      content={({ payload }) => (
                        <div style={{ display: 'flex', gap: 18, padding: 0, margin: 0 }}>
                          {payload &&
                            payload.map((entry: any, idx: number) => (
                              <div key={`legend-${idx}`} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <div
                                  style={{
                                    width: 15,
                                    height: 15,
                                    borderRadius: 4,
                                    background: entry.color,
                                    border: '2px solid #111827',
                                    marginRight: 6,
                                    boxSizing: 'border-box',
                                    display: 'inline-block',
                                  }}
                                />
                                <span
                                  style={{
                                    color: '#111827',
                                    fontWeight: 400,
                                    fontSize: 13,
                                    fontFamily: 'inherit',
                                    WebkitFontSmoothing: 'auto',
                                    MozOsxFontSmoothing: 'auto',
                                  }}
                                >
                                  {entry.value}
                                </span>
                              </div>
                            ))}
                        </div>
                      )}
                    />
                    <Bar dataKey="focusedMin" stackId="a" fill="#111827" stroke="#111827" name="Focused (min)" />
                    <Bar dataKey="distractedMin" stackId="a" fill="#FFFFFF" stroke="#111827" name="Distracted (min)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="focusdash-card-hint">Showing the last 12 sessions.</div>
            </div>
          </div>

          <div className="focusdash-card focusdash-tablecard">
            <div className="focusdash-card-title">Recent sessions</div>
            <div className="focusdash-tablewrap">
              <table className="focusdash-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Course</th>
                    <th>Session</th>
                    <th>Focused</th>
                    <th>Distracted</th>
                    <th>Focus %</th>
                    <th>Distractions</th>
                    <th>Device</th>
                  </tr>
                </thead>
                <tbody>
                  {latestRows.map((s) => {
                    const end = s.endTs ? new Date(s.endTs * 1000) : (s.createdAt ?? null);
                    const courseName = s.courseId ? (courseNameById.get(s.courseId) ?? s.courseId.slice(0, 8) + '…') : '—';
                    const linkedSessionId = s.sessionId ?? s.courseSessionId ?? null;
                    const sessionName = linkedSessionId ? (sessionNameById.get(linkedSessionId) ?? linkedSessionId.slice(0, 8) + '…') : '—';
                    const focusedMin = Math.round(msToMinutes(s.focusedMs));
                    const distractedMin = Math.round(msToMinutes(s.distractedMs));
                    const focusPct = s.focusPercent == null ? '—' : `${Math.round(s.focusPercent)}%`;
                    const deviceShort = s.deviceId ? s.deviceId.slice(0, 8) + '…' : '—';
                    return (
                      <tr key={s.id}>
                        <td>{end ? formatDateTime(end) : '—'}</td>
                        <td>{courseName}</td>
                        <td>{sessionName}</td>
                        <td>{focusedMin}m</td>
                        <td>{distractedMin}m</td>
                        <td>{focusPct}</td>
                        <td>{s.distractions ?? 0}</td>
                        <td>{deviceShort}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="focusdash-card-hint">
              Tip: focus values come from the Pi agent’s smoothed focused/distracted state.
            </div>
          </div>
        </>
      )}
    </div>
  );
}

