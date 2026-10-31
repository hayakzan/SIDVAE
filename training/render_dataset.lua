-- chipsynth C64 dataset generator (Osc1 wave + pulse width + pre-roll)
-- ReaScript (Lua) for REAPER
--
-- First run: set PROBE_PARAMS = true, run once, and find Osc 1's pulse-width
-- parameter in the console. Put its index in PW_PARAM_IDX, set PROBE_PARAMS back
-- to false, and run again to render.

-- ====== USER CONFIG ======
local OUT_DIR       = [[/Users/hayakzan/Desktop/JUCE/SIDVaePlugin/SIDVae/training/data]]
local NOTE_LEN      = 2.0
local SILENCE_LEAD  = 0.20

-- Every semitone from A1 (55 Hz) to A6 (1760 Hz): the model needs to see pitch
-- vary smoothly to learn anything from log_f0.
local MIDI_LO, MIDI_HI, MIDI_STEP = 33, 93, 1

-- Three levels: enough to test whether velocity changes the timbre or only the
-- level (training normalizes level away).
local VEL_LEVELS = {60, 90, 120}

-- Noise is excluded from training, so it's off by default.
local WAVES         = {"triangle", "saw", "pulse"}

-- Osc 1 Wave is a bitfield (like the SID's waveform bits): value * 8 =
-- 1 Tri, 2 Saw, 4 Pulse, 8 Noise. Values in between are combined waveforms.
local ENUM_WAVE_PARAM = 7
local ENUM_WAVE_VALUES = {
  triangle = 0.125,
  saw      = 0.250,
  pulse    = 0.500,
  noise    = 1.000
}

-- Osc 1 Pulse Width: index from the probe run. Appears in this format: 
-- [  5] Oscillator 1 : PW = 2048 so 5 went in in this case
-- Rendering pulse stops with an error while this is -1
local PW_PARAM_IDX = 5
-- Osc 1 pulse width as SID register values (0-4095). Duty = value / 4096
-- Only up to 50%: a 70% pulse is a 30% pulse inverted and sounds the same.
-- The console prints what chipsynth shows for each- it should match these.
local PW_REGS = {205, 410, 819, 1229, 1638, 2048}

-- true = list chipsynth's parameters in the console and stop (no rendering)
-- set false once you get the value 
local PROBE_PARAMS = false

-- ====== UTIL ======
local r = reaper
local function log(s) r.ShowConsoleMsg(tostring(s).."\n") end

local function ensure_dir_exists(path)
  r.RecursiveCreateDirectory(path, 0) -- ok if exists
  local probe = path .. "/.write_test"
  local f = io.open(probe, "w")
  if not f then log("ERROR: cannot write to OUT_DIR: "..path); return false end
  f:write("ok"); f:close(); os.remove(probe); return true
end

local function freq_of_midi(n) return 440.0 * (2 ^ ((n - 69)/12.0)) end
local function get_track(idx) return r.GetTrack(0, idx) end
local function find_fx_on_track(tr, needle)
  for i=0,r.TrackFX_GetCount(tr)-1 do
    local _, name = r.TrackFX_GetFXName(tr,i,"")
    if name:lower():find(needle,1,true) then return i, name end
  end
  return -1,nil
end

local function probe_params(tr, fx)
  local n = r.TrackFX_GetNumParams(tr, fx)
  log(string.format("%d parameters. Showing the first 60 and any that mention pulse/width/PW:", n))
  for i = 0, n - 1 do
    local _, name = r.TrackFX_GetParamName(tr, fx, i, "")
    local lname = name:lower()
    local match = lname:find("pulse") or lname:find("width") or lname:find("pw") or lname:find("duty")
    if (i < 60 or match) and not lname:find("^ch%d+ %- cc") then
      local _, shown = r.TrackFX_GetFormattedParamValue(tr, fx, i, "")
      log(string.format("  [%3d] %-40s = %s", i, name, shown))
    end
  end
end

local function set_enum_wave(tr, fx, wave)
  local v = ENUM_WAVE_VALUES[wave]; if not v then error("Unknown wave: "..tostring(wave)) end
  r.TrackFX_SetParamNormalized(tr, fx, ENUM_WAVE_PARAM, v)
  r.TrackFX_EndParamEdit(tr, fx, ENUM_WAVE_PARAM)
  local _, shown = r.TrackFX_GetFormattedParamValue(tr, fx, ENUM_WAVE_PARAM, "")
  log(string.format("wave %s -> chipsynth shows %s", wave, shown))
end

-- Sets the pulse width and reads it back, so a wrong index shows up in the console.
local function set_pw(tr, fx, pw_norm)
  r.TrackFX_SetParamNormalized(tr, fx, PW_PARAM_IDX, math.max(0, math.min(1, pw_norm)))
  r.TrackFX_EndParamEdit(tr, fx, PW_PARAM_IDX)
  local got = r.TrackFX_GetParamNormalized(tr, fx, PW_PARAM_IDX)
  local _, shown = r.TrackFX_GetFormattedParamValue(tr, fx, PW_PARAM_IDX, "")
  if math.abs(got - pw_norm) > 0.01 then
    log(string.format("WARNING: pulse width set to %.3f but reads back %.3f", pw_norm, got))
  end
  return shown
end

local function make_fresh_midi_item(track, t_end, t0, t1, pitch, vel)
  -- clear previous items so we always have a real MIDI item
  for i=r.CountTrackMediaItems(track)-1,0,-1 do
    r.DeleteTrackMediaItem(track, r.GetTrackMediaItem(track,i))
  end
  local item = r.CreateNewMIDIItemInProj(track, 0, t_end, false)
  local take = r.GetActiveTake(item)
  local ppq0 = r.MIDI_GetPPQPosFromProjTime(take, t0)
  local ppq1 = r.MIDI_GetPPQPosFromProjTime(take, t1)
  r.MIDI_InsertNote(take, false,false, ppq0, ppq1, 0, pitch, vel, false)
  r.MIDI_Sort(take)
  return item, take
end

-- ====== MAIN ======
r.ShowConsoleMsg("")

local track = get_track(0); if not track then log("ERROR: No Track 1."); return end
local fx_idx, fx_name = find_fx_on_track(track, "chipsynth"); if fx_idx<0 then log("ERROR: chipsynth not found."); return end
log("Using FX: "..fx_name)

if PROBE_PARAMS then probe_params(track, fx_idx); return end

local midi_notes = {}
for m = MIDI_LO, MIDI_HI, MIDI_STEP do midi_notes[#midi_notes + 1] = m end

local uses_pulse = false
for _, w in ipairs(WAVES) do if w == "pulse" then uses_pulse = true end end
if uses_pulse and PW_PARAM_IDX < 0 then
  log("ERROR: PW_PARAM_IDX is not set. Run once with PROBE_PARAMS = true to find it.")
  return
end

-- Count the renders and ask before starting.
local total = 0
for _, w in ipairs(WAVES) do
  total = total + #midi_notes * #VEL_LEVELS * ((w == "pulse") and #PW_REGS or 1)
end
local msg = string.format("%d files of %.1f s (%d notes, %d velocities, %d pulse widths).\n\nStart rendering?",
  total, SILENCE_LEAD + NOTE_LEN, #midi_notes, #VEL_LEVELS, uses_pulse and #PW_REGS or 0)
if r.ShowMessageBox(msg, "chipsynth dataset", 1) ~= 1 then log("Cancelled."); return end

if not ensure_dir_exists(OUT_DIR) then return end
r.Undo_BeginBlock()

local manifest_path = OUT_DIR.."/manifest.csv"
local mf = io.open(manifest_path,"w"); mf:write("file,wave,midi_note,freq_hz,velocity,pw_norm\n")

local done = 0
for _, wave in ipairs(WAVES) do
  set_enum_wave(track, fx_idx, wave)

  local pw_list = (wave == "pulse") and PW_REGS or {false}
  for _, pw in ipairs(pw_list) do
    if pw then log(string.format("PW %d -> chipsynth shows %s", pw, set_pw(track, fx_idx, pw / 4095))) end

    for _, midi in ipairs(midi_notes) do
      local hz = freq_of_midi(midi)
      for _, vel in ipairs(VEL_LEVELS) do
        local t0 = SILENCE_LEAD
        local t1 = SILENCE_LEAD + NOTE_LEN
        local t_end = t1
        r.GetSet_LoopTimeRange(true,false, 0, t_end, false) -- render window
        r.Main_OnCommand(40345,0) -- all notes off

        -- create a real MIDI item and insert the note
        make_fresh_midi_item(track, t_end, t0, t1, midi, vel)

        -- tag = duty cycle in percent (205 -> pw05, 2048 -> pw50)
        local pw_tag = pw and string.format("_pw%02d", math.floor(pw / 4096 * 100 + 0.5)) or ""
        local fname = string.format("%s_%s_m%03d_%0.1fHz_vel%d%s",
                        wave, "chipsynth", midi, hz, vel, pw_tag)
        local full_no_ext = OUT_DIR.."/"..fname
        r.GetSetProjectInfo_String(0,"RENDER_FILE",full_no_ext,true)
        r.Main_OnCommand(41824,0) -- render

        -- pw_norm column = duty cycle (0..0.5)
        mf:write(string.format("%s.wav,%s,%d,%.4f,%d,%s\n",
          fname, wave, midi, hz, vel, pw and string.format("%.3f", pw / 4096) or ""))

        done = done + 1
        if done % 25 == 0 then log(string.format("%d / %d", done, total)) end
        r.UpdateArrange()
      end
    end
  end
end

mf:close()
r.Undo_EndBlock("Generate chipsynth C64 dataset (Osc1 wave)", -1)
log(string.format("Done. %d files and manifest written to: %s", done, OUT_DIR))
